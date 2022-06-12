from gym.envs.registration import register
from .envs.auction_market import pre_run_slot_generator, Observation

register(
    id='AuctionEmulator-v0',
    entry_point='auction_simulator.envs:AuctionEmulatorEnv',
)

register(
    id='YewuSimulator_train-v0',
    entry_point='auction_simulator.envs:Simulator',
    kwargs=dict(mode='train', cfg=None, data_folder=None)
)
register(
    id='YewuSimulator_test-v0',
    entry_point='auction_simulator.envs:Simulator',
    kwargs=dict(mode='test', cfg=None, data_folder=None)
)

register(
    id='SynSimulator_train-v0',
    entry_point='auction_simulator.envs:Simulator',
    kwargs=dict(mode='train', cfg=None, data_folder=None)
)
register(
    id='SynSimulator_test-v0',
    entry_point='auction_simulator.envs:Simulator',
    kwargs=dict(mode='test', cfg=None, data_folder=None)
)