import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import matplotlib.pyplot as plt  # Show the impact of frame stacking
import os
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


# The SIMPLE_MOVEMENT will simplify the character's actions, facilitating the algorithm learn how to play

CHECKPOINT_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/train'
LOGS = str(pathlib.Path(__file__).parent.resolve()) + '/logs'


class App:
    def __init__(self):
        """
        Setting up our game environment and start
        """
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
        self.preproc()
       # self.training_model()

    def run_game(self):
        """
        Function that will run the game
        :return:
        """
        # Loading model
        self.model = PPO.load('best_model20000.zip')
        state = self.env.reset()
        while True:
            action, _state = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            self.env.render()

    def preproc(self):
        """
        Function to preprocess game and convert it to gray scale, reducing the data to AI process
        :return:
        """

        # Converting env into gray scale
        self.env = GrayScaleObservation(self.env, keep_dim=True)

        # Wrapping into dummy enviromnment
        self.env = DummyVecEnv([lambda: self.env])

        # Stacking frames
        self.env = VecFrameStack(self.env, 4, channels_order='last')

    def training_model(self):
        """
        Function who will train our ML model
        :return:
        """
        # AI model started
        self.model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log=LOGS, learning_rate=0.000001, n_steps=10)
        self.model.learn(total_timesteps=1000, callback=self.callback)


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = str(pathlib.Path(__file__).parent.resolve()) + f'/best_model{self.n_calls}'
            self.model.save(model_path)

        return True


