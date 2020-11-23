import gym
import cv2
import numpy as np
from agents import Agent
from torch.utils.tensorboard import SummaryWriter
import os


if __name__ == "__main__":
    _env = gym.make("LunarLander-v2")
    print(_env.observation_space)
    print(_env.action_space)

    _d_log = "logs"
    _f_info = "_info.log"
    _f_checkpoint = "_checkpoint"

    _episode = 0
    _scores = []
    if os.path.exists(_f_info):
        with open(_f_info, "r") as _f:
            for _line in _f.readlines():
                _a, _b = _line.split("\t")
                _episode = int(_a)
                _scores.append(float(_b))
    _n_games = _episode + 3000

    _agent = Agent((8,), 4)
    if os.path.exists(_f_checkpoint):
        _agent.net.load_checkpoint(_f_checkpoint)

    _writer = SummaryWriter(_d_log)
    _is_quit = False
    while _episode < _n_games:
        _observation = _env.reset()
        _done = False
        _score = 0.0
        while not _done:
            _action = _agent.get_action(_observation)
            _next_observation, _reward, _done, _info = _env.step(_action)
            _score += _reward
            _agent.learn(_observation, _reward, _next_observation, _done)
            _observation = _next_observation

            _rgb = _env.render("rgb_array")
            _bgr = cv2.cvtColor(_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", _bgr)
            _key_code = cv2.waitKey(1)
            if _key_code in [27, ord('q')]:
                _is_quit = True
                break
        if _is_quit:
            break
        _scores.append(_score)
        _episode += 1
        _avg_score = float(np.mean(_scores[-100:]))
        if _episode % 500 == 0:
            _agent.net.save_checkpoint(_f_checkpoint + "-%d" % _episode)

        _writer.add_scalar('ActorCritic/score', _score, _episode)
        _writer.add_scalar('ActorCritic/avg_score', _avg_score, _episode)
        print("Episode: %d score: %.2f avg: %.2f" % (_episode, _score, _avg_score))
        with open(_f_info, "a+") as _f:
            _f.write("%d\t%.2f\n" % (_episode, _score))
    _agent.net.save_checkpoint(_f_checkpoint)
    print("Closing...")
    _env.close()
