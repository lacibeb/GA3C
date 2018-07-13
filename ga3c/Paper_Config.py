# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

class Config:

    #########################################################################
    # Game configuration

    # Name of the game, with version (e.g. PongDeterministic-v0)
    # GAME = 'PongDeterministic-v0'
    GAME = 'Pyperrace'

    STATE_DIM = 4
    ACTION_DIM = 1

    # Enable to see the trained agent in action
    SHOW_WINDOW = False
    # Enable to train
    TRAIN_MODELS = True

    #########################################################################
    # Algorithm parameters
    
    # steps per game
    MAX_STEPS = 5
    
    # Reward Clipping
    REWARD_MIN = -100
    # REWARD_MAX = 1

    track_name = 'h1'
    ref_calc = 'default'
    car_name = 'Touring'
    random_init = False
    save_env_ref_buffer_dir = './env_ref_buffer'
    save_env_ref_buffer_name = 'env_ref_buffer_1'
    load_env_ref_buffer = './env_ref_buffer/env_ref_buffer_1'
    load_all_env_ref_buffer_dir = './env_ref_buffer'

    reward_based_on_ref = False

    logging_game = False
    logging_debug = False
    logging_step = False

