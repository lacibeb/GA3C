# This stores trafck definitions

import numpy as np


def get_track_params(track_name):
    # Setting paramters for given track name
    # adding new track:
    # always add startline, endline, sections, track_file, trk_col, ref_action for a track
    # sections is not necessary
    if track_name == 'h1':
        # as hungaroring turn 1
        startline = np.array([200, 220, 200, 50])
        endline = np.array([200, 1250, 250, 1400])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_outside_color = np.array([255, 255, 255], dtype='uint8')
        track_file = 'h1.bmp'
        sections = []
        ref_action = ([70, -70, 0, -180, -180, -180, -165, -150, -140, -120, -110, -100, -90, -90, -80, -80, -80, -80,
                         -40, -40, -30, -30, -20, -20, -20, -20, -10, -30])

    elif track_name == 'PALYA3':
        # palya 3-hoz
        sections = np.array([[350,  60, 350, 100],
                        [425, 105, 430, 95],
                        [500, 140, 530, 110],
                        [520, 160, 580, 150]])

        startline = np.array([ 35, 200,  70, 200])
        endline = np.array([250,  60, 250, 100])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_outside_color = np.array([255, 255, 255], dtype='uint8')
        track_file = 'PALYA3.bmp'
        ref_action = []

    elif track_name == 'palya4':
        # palya4 teljes
        startline = np.array([273, 125, 273, 64])
        endline = np.array([100, 250, 180, 250])
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_outside_color = np.array([255, 255, 255], dtype='uint8')
        track_file = 'PALYA4.bmp'
        sections = []
        ref_action = ([0, -180, -96, -97, -110, -105, -105, -105, -110, 110, 110, 100, 50,
                       150, 150, 90, 40, -140, -100, -120, -120, -120, -65, -20, -25, -80,
                       -110, -110, -110, -95])
        # van egy referencia lepessor ami valahogy beér a célba (palya5) :
        # self.ref_actions = np.array

        # h1.bmp-hez:
        # #ember által lejátszott lépések: (


    elif  track_name == 'palya5':
        # palya5.bmp-hez:
        startline = np.array([670, 310, 670, 130])  # [333, 125, 333, 64],[394, 157, 440, 102],
        endline = None
        trk_col = np.array([99, 99, 99])  # pálya színe (szürke)
        track_inside_color = np.array([255, 0, 0], dtype='uint8')
        track_outside_color = np.array([255, 255, 255], dtype='uint8')
        track_file = 'PALYA5.bmp'
        sections = []
        ref_action = ([0, 150, 180, -160, -160, -160, -150, -90, -90, -110, -110, -120, -110, -110, 0,90, -90, 90,
                       -140, 90, 110, 90, 120, 120, 120, 120, 100, -20, -10, 0, 0, 0, 0])
    else:
        raise ValueError('With this name no track is found')

    return track_file, trk_col, track_inside_color, track_outside_color, startline, endline, sections, ref_action

def get_side_sections(track_name, error):
    if track_name == 'h1':
        # as hungaroring turn 1
    if error == 3:
        return [[201, 58, 636, 58], [636, 58, 1034, 54], [1034, 54, 1376, 52], [1376, 52, 1447, 71], [1447, 71, 1534, 121],
             [1534, 121, 1577, 152], [1577, 152, 1600, 183], [1600, 183, 1633, 255], [1633, 255, 1651, 345],
             [1651, 345, 1647, 461], [1647, 461, 1607, 563], [1607, 563, 1581, 605], [1581, 605, 1503, 676],
             [1503, 676, 1433, 733], [1433, 733, 1335, 807], [1335, 807, 1214, 893], [1214, 893, 1138, 959],
             [1138, 959, 1106, 983], [1106, 983, 1032, 1033], [1032, 1033, 967, 1075], [967, 1075, 885, 1123],
             [885, 1123, 803, 1165], [803, 1165, 741, 1196], [741, 1196, 600, 1264], [600, 1264, 529, 1292],
             [529, 1292, 384, 1342], [384, 1342, 306, 1367], [306, 1367, 226, 1390], [226, 1390, 26, 1427],
             [26, 1427, 1, 1429]], \
            [[191, 184, 1322, 187], [1322, 187, 1390, 204], [1390, 204, 1439, 235], [1439, 235, 1475, 281],
             [1475, 281, 1493, 333], [1493, 333, 1498, 395], [1498, 395, 1492, 470], [1492, 470, 1472, 526],
             [1472, 526, 1437, 570], [1437, 570, 1307, 667], [1307, 667, 1272, 691], [1272, 691, 1179, 757],
             [1179, 757, 1151, 780], [1151, 780, 1113, 809], [1113, 809, 1050, 864], [1050, 864, 981, 909],
             [981, 909, 928, 945], [928, 945, 893, 968], [893, 968, 821, 1010], [821, 1010, 739, 1052],
             [739, 1052, 531, 1153], [531, 1153, 353, 1216], [353, 1216, 264, 1244], [264, 1244, 191, 1265],
             [191, 1265, 24, 1296], [24, 1296, 0, 1297]]

    return None, None

def get_ref_actions(track_name, car_name):
    if track_name == 'h1':
        if car_name == 'Touring':
            # as hungaroring turn 1
            stored_actions = ([70, -70, 0, -180, -180, -180, -165, -150, -140, -120, -110, -100, -90, -90, -80, -80, -80, -80, -40, -40, -30,
             -30, -20, -20, -20, -20, -10, -30], \
            [90, -90, -50, 100, -180, -180, -180, -160, -150, -140, -110, -100, -90, -90, -80, -70, -70, -70, -60, -40, -20,
             -20, -20, -20, -10, -10, -10], \
            [-90, 90, 50, -100, -180, -180, -180, -160, -150, -140, -110, -100, -90, -90, -80, -70, -70, -70, -40, -30, -20,
             -20, -20, -20, -20, -20], \
            [70, -70, 0, -180, -180, -180, -165, -155, -150, -130, -110, -100, -90, -90, -80, -70, -60, -60, -40, -40, -30,
             -20, -20, -20, -20, -20], \
            [90, -40, -10, -180, -180, -180, -175, -165, -150, -120, -100, -90, -70, -70, -70, -70, -70, -70, -70, -60, -50,
             -40, -40, -30, -30, -20], \
            [-180, -180, -180, -180, 0, 80, -80, 0, 0, -175, -165, -150, -150, -100, -90, -70, -70, -70, -80, -80, -70, -50, -50, -40,
             -30, -30, -30, -30, -20], \
            [-180, -150, 150, 150, 90, -90, 0, -90, 130, -120, -120, -110, -100, -100, -100, -100, -100, -90, -80, -80, -60,
             -40, -40, -40, -20, -20, -20, -20, -20], \
            [-180, -150, -90, 90, 90, 120, 120, -130, -120, -120, -120, -100, -110, -100, -100, -100, -100, -90, -80, -70,
             -40, -40, -20, -20, -20, -20, -20, -20, -20], \
            [0, -180, -180, -170, 120, 120, -130, -130, -130, -100, 100, 100, 100, -110, -110, -110, -110, -110, -100, -100,
             100, 100, -100, -100, -120, -100, -100, 90, 80, -60, 60, -30, -30, -20, -30, -30, -40, -40, -50, -30])
        elif car_name == 'Gokart':
            stored_actions = ([-180, -180, -180, -180, -180, 110, -110, 110, -110, 20, 30, -30, -110, -110, 110, 110,\
                              -110, -110, -110, -110, -110, -110, -90, 90, 130, -110, -11, -110, -110, -110, 110, -70,\
                              70, -70, 40, 20, -20, -20, -30, -60, 20, -30, -90])
        #end Hung turn 1

    elif track_name == 'PALYA3':
        # palya 3-hoz
        stored_actions = None

    elif track_name == 'palya4':
        # palya4 teljes
        stored_actions = None


    elif  track_name == 'palya5':
        # palya5.bmp-hez:
        stored_actions = None


    else:
        raise ValueError('With this name no track is found')

    return np.array(stored_actions)




