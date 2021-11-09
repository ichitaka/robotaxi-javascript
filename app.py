# base imports
import datetime
import pickle
import copy
from typing import List, Optional
import logging
import os
import uuid
import random
import json

# package imports
from fastapi import FastAPI, Request, status, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import socketio
from pymongo import MongoClient
from starlette.responses import Response
import cv2
import asyncio
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from dotenv import find_dotenv, load_dotenv

# local imports
from robogame import RoboTaxi

# load env variables
load_dotenv(find_dotenv())

# setup database
client = MongoClient(os.environ['MONGODB_URI'])
db = client.final_experiment
collection = db['experiment_data']

# setup variables for WebRTC connection
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# setup server
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
origins = [
    "http://localhost",
    "http://localhost:5000",
    "https://localhost",
    "https://localhost:5000",
]
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# game globals
game = None
# needed to stop render loop when client disconnects
active_session = False
# if the self played session should be recorded and saved as a pickle file
record_session = False
# This is used to record sessions
memory = {
    'initial_parameters': None,
    'steps': []
}
# The state of the current running session
current_session = None
# If there is a session running right now
there_is_a_running_session = False
# At which time the session started
last_session_update_time = None


class Session():
    '''This class is used to manage a single experiment session.'''
    def __init__(self, subject):
        self.run = 0
        self.experiment_step = 0
        self.score = 30
        self.step = 0
        self.subject = subject
        self.session_order = get_subject_experiment_order()
        self.condition = get_condition()
        save_in_database(subject, {'condition': self.condition})
        save_in_database(subject, {'session_order': self.session_order})

    def next_experiment_step(self, move=0):
        experiment_step = self.experiment_step + move
        print('Calling next step: ', experiment_step)
        if len(self.session_order) > experiment_step:
            next_step = self.session_order[experiment_step]
        else:
            next_step = self.session_order[-1]
        return next_step

    def progress_step(self):
        self.experiment_step += 1


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that we use to process the incoming data with.
    """
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.subject = current_session.subject
        os.makedirs(f'data/{self.subject}', exist_ok=True)
        self.run = current_session.run

    async def recv(self):
        frame = await self.track.recv()
        current_step = current_session.step
        time = datetime.datetime.now().isoformat()
        filename = (f'data/{self.subject}'
                    f'/{self.subject}'
                    f'_{self.run}'
                    f'_{current_step}'
                    f'_{time}.jpg')
        img = frame.to_ndarray(format="bgr24")
        cv2.imwrite(filename, img)
        return frame


@app.on_event("shutdown")
async def shutdown_event():
    '''To end the WebRTC connection.'''
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


@app.post('/offer')
async def offer(request: Request):
    '''Function for the setup of a WebRTC connection.'''
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        pc.addTrack(
            VideoTransformTrack(
                relay.subscribe(track),
                transform=params["video_transform"]
            )
        )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


class UserData(BaseModel):
    '''This class is used to define the variables that will be stored in
    the database.'''
    agreement_1: Optional[bool]
    agreement_2: Optional[bool]
    agreement_3: Optional[bool]
    agreement_time: Optional[str]
    survey1: Optional[int]
    survey2: Optional[int]
    survey3: Optional[int]
    survey4: Optional[int]
    survey5: Optional[int]
    survey6: Optional[str]
    survey7: Optional[int]
    subject_code: Optional[str]


class GameInput(BaseModel):
    '''This class defines the data structure of the robotaxi
    game controls we receive.'''
    next_transition: List[int]


class RecordedRoboTaxi:
    '''This class is used to replay a previously recorded session.'''
    def __init__(self, session_path, score=0) -> None:
        data = json.load(open(session_path, 'r'))
        self.steps = data['steps']
        self.start_parameters = data['initial_parameters']
        self.current_step = 0
        self.score = score

    def update(self, transition):
        pass

    def get_render(self):
        if len(self.steps) > self.current_step:
            current_render = self.steps[self.current_step]
            self.current_step += 1
        return current_render

    def initial_parameters(self):
        params = self.start_parameters
        return params


def validate(key, data):
    '''This function is used to validate the data received from
     the client. We haven't used it much, but for further
     development it might be useful.
    Returns: True if the data is valid, False otherwise.'''
    if key in ['agreement_1', 'agreement_2']:
        if data == '':
            return False
        else:
            return True
    if key == 'subject_code':
        if len(data) != 6:
            return False
        else:
            return True
    return True


def get_condition():
    '''This function is used to get the condition of the current session.
    The conditions are always equally distributed.'''
    condition1_count = collection.count_documents({'condition': 1})
    condition2_count = collection.count_documents({'condition': 2})
    if condition1_count > condition2_count:
        return 2
    else:
        return 1


def reset_current_session():
    '''Unblocks a running session.'''
    global there_is_a_running_session
    global last_session_update_time
    global current_session
    there_is_a_running_session = False
    last_session_update_time = None
    current_session = None


def ping_current_session():
    '''Adds a current timestamp to the running session to further
    block the active session from new participants.'''
    global last_session_update_time
    last_session_update_time = datetime.datetime.now()


def generate_database_document(subject_code):
    '''Inserts a document into the database. If a subject with the same
    subject code is already available, adds a count variable to the
    subject code.'''
    query = collection.count_documents({'subject_code': subject_code})
    if query > 0:
        print('subject code already exists')
        subject_code = subject_code + f'-{query + 1}'
    return subject_code


def save_in_database(subject_code, data):
    '''Updates the document in the database with the newly incoming data.'''
    if data:
        collection.update_one({'subject_code': subject_code},
                              {'$set': data},
                              upsert=True)
    else:
        print('Received empty data')


def get_subject_experiment_order():
    '''Generates the random order of the recorded sessions.'''
    full = [0, 1, 8]
    only_positive = [20, 21, 23, 24]
    runs = []
    for layout in full:
        for pattern in range(7):
            runs.append(f'layout_{layout}_pattern_{pattern}')
    for pattern in range(5):
        runs.append(f'layout_14_pattern_{pattern}')
    for layout in only_positive:
        runs.append(f'layout_{layout}_pattern_0')
    random.shuffle(runs)
    experiment_order = runs
    experiment_order.insert(0, 'session_0')
    experiment_order.insert(len(experiment_order), 'end')
    return experiment_order


@app.post('/form/', status_code=200)
async def form_data(data: UserData,
                    response: Response,
                    subject_code: Optional[str] = Cookie(None)):
    '''This function is used to handle the data received from
    the client. It is called when the client submits the form.
    The data is validated and added to the database if valid.
    Otherwise an error message is displayed.'''
    data = data.dict()
    data = {k: v for k, v in data.items() if v is not None}
    for key in data.keys():
        if data.get(key):
            valid = validate(key, data[key])
            if not valid:
                print(f'Issue with {key}')
                response.status_code = status.HTTP_406_NOT_ACCEPTABLE
                return {'alert': 'Etwas stimmt mit Ihrer Eingabe nicht'}
    if data.get('subject_code'):
        subject_code = data.get('subject_code')
        subject_code = generate_database_document(subject_code)
        data['subject_code'] = subject_code
        response.set_cookie(key='subject_code', value=subject_code)
        global there_is_a_running_session
        global last_session_update_time
        there_is_a_running_session = True
        last_session_update_time = datetime.datetime.now()
    else:
        if subject_code:
            print(subject_code)
        else:
            response.status_code = status.HTTP_406_NOT_ACCEPTABLE
            return {
                'alert': ('VP-Code fehlt!'
                          'Gehen sie zurück auf'
                          ' <a href="/">tu-darmstadt-experiment.xyz</a>.')
                }
    save_in_database(subject_code, data)
    return {''}


@app.get('/init')
def initialize():
    '''This function is used to return the initial parameters
    of the currently running game.'''
    global memory
    parameters = game.initial_parameters()
    run = current_session.run
    if record_session or run == 0:
        memory['initial_parameters'] = copy.deepcopy(parameters)
    parameters['score'] = current_session.score
    return parameters


@sio.on('render')
def render(sid):
    '''Starts the background task, that returns the individual
    frames of the robotaxi game to the client, through Websockets.'''
    global active_session
    active_session = True
    current_session.step = 0
    sio.start_background_task(target=send_render_data)


async def send_render_data():
    '''Sends the frames of the running game to the client.'''
    global memory
    global current_session
    run = current_session.run
    while active_session:
        game_render = game.get_render()
        current_session.step += 1
        if record_session or run == 0:
            memory['steps'].append(copy.deepcopy(game_render))
        if game_render.get('end'):
            await sio.emit('render', game_render)
            break
        else:
            game_render['score'] += current_session.score
            await sio.sleep(1/12)
            await sio.emit('render', game_render)


@sio.on('connect')
def connected(sid, two, three):
    '''Connection function for the WebSocket connection.'''
    print('connected')


@sio.on('disconnect')
def disconnect(sid):
    '''Disconnection function for the WebSocket connection.'''
    global active_session
    active_session = False
    print('disconnect ', sid)


@app.post('/inputs')
def recieve_inputs(input: GameInput):
    '''POST request function to receive the keyboard inputs of the clients.'''
    if game:
        game.update(input.next_transition)


@app.get('/end')
def end(score: int, subject_code: Optional[str] = Cookie(None)):
    '''This function is always called, when the client received
    the last frame of the game. It updates the database with the
    score and the end time of the session. After that it determines
    the next step of the experiment.'''
    if current_session.run > 0:
        current_session.score = score
    else:
        save_in_database(current_session.subject,
                         {'user_session': copy.deepcopy(memory)})
    save_in_database(subject_code,
                     {f'run_{current_session.run}_end_time':
                         datetime.datetime.now()
                      })
    print('Finished Run:', current_session.run)
    current_session.run += 1
    run = current_session.run
    if run > 30:
        save_in_database(subject_code, {'final_score': current_session.score})
        target_url = '/post-game'
    else:
        target_url = '/car-selection'
    if record_session:
        pickle.dump(memory,
                    open(f'trial_{datetime.datetime.now().isoformat()}.pickle',
                         'wb')
                    )
    return {'SUCCESS': True, 'target': target_url}


@app.get('/car-selection')
def car_selection(request: Request,
                  subject_code: Optional[str] = Cookie(None)):
    global current_session
    ping_current_session()
    if not current_session:
        current_session = Session(subject_code)
        print(current_session.session_order)
    return templates.TemplateResponse(
        "car-selection.html",
        {"request": request,
         'run': current_session.run,
         'score': current_session.score,
         })


@app.get('/game')
def game_page(request: Request,
              car: str,
              session: Optional[str] = None,
              subject_code: Optional[str] = Cookie(None)):
    global game
    global current_session
    ping_current_session()
    run = current_session.run
    next_step = current_session.next_experiment_step()
    current_session.progress_step()
    if run > 0 and car:
        current_session.score -= 1  # pay for car
        save_in_database(subject_code, {f'car_{run}': car})
    save_in_database(subject_code,
                     {f'run_{run}_start_time': datetime.datetime.now()})
    if run == 0:
        car = 'auto_bus'
        game = RoboTaxi(max_steps=20, num_obj=1, add_new=False)
        user_plays = True
        global memory
        memory = {
            'initial_parameters': None,
            'steps': []
        }
        save_in_database(subject_code, {f'car_{run}': car})
    else:
        current_session.step = -1
        if session:
            next_step = session
        game = RecordedRoboTaxi(session_path=f'sessions/{next_step}.json',
                                score=current_session.score
                                )
        user_plays = False
    return templates.TemplateResponse("robotaxi_game.html",
                                      {"car": car, "request": request,
                                       "user_plays": user_plays,
                                       "run": run
                                       }
                                      )


@app.get('/survey')
def survey(request: Request):
    return templates.TemplateResponse('survey.html', {'request': request})


@app.get('/post-game')
def post_game(request: Request, subject_code: Optional[str] = Cookie(None)):
    score = current_session.score
    reset_current_session()
    return templates.TemplateResponse('post-game.html',
                                      {'request': request, 'score': score})


@app.route('/camera-setup', methods=['GET', 'POST'])
def game_intro(request: Request):
    ping_current_session()
    return templates.TemplateResponse("camera-setup.html",
                                      {"request": request})


@app.get('/game-instructions')
def instructions(request: Request, condition: int = None,
                 subject_code: Optional[str] = Cookie(None)):
    global current_session
    if not current_session:
        current_session = Session(subject_code)
    if not condition:
        ping_current_session()
        condition = current_session.condition
    return templates.TemplateResponse('game-instructions.html',
                                      {'request': request,
                                       'condition': condition})


@app.route('/agreement', methods=['GET', 'POST'])
def agreement(request: Request):
    ping_current_session()
    return templates.TemplateResponse("agreement.html",
                                      {"request": request}
                                      )


@app.route('/subject-code', methods=['GET', 'POST'])
def subject_code(request: Request):
    if there_is_a_running_session:
        time_passed = datetime.datetime.now() - last_session_update_time
        if time_passed < datetime.timedelta(minutes=5):
            return templates.TemplateResponse('session-already-running.html',
                                              {'request': request})
        else:
            reset_current_session()
            return templates.TemplateResponse("subject-code.html",
                                              {"request": request})
    else:
        reset_current_session()
        return templates.TemplateResponse("subject-code.html",
                                          {"request": request})


@app.route('/study-end', methods=['GET', 'POST'])
def study_end(request: Request):
    return templates.TemplateResponse("study-end.html",
                                      {"request": request})


@app.route('/', methods=['GET', 'POST'])
def index(request: Request):
    return templates.TemplateResponse("index.html",
                                      {"request": request}
                                      )


@app.route('/test', methods=['GET'])
def test(request: Request):
    return templates.TemplateResponse("test.html",
                                      {"request": request}
                                      )


app.mount('/', socket_app)
