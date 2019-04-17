import pandas as pd
import pickle as pick


# Method to grab the data from a given csv file (input will only find files in the csv_files folder and only
# files containing the file type .csv), drops the necessary collumns and returns the raw data as a pandas data frame
def parse_data():
    file_name = ""
    df = pd.DataFrame
    incorrect_file = True
    while incorrect_file:
        try:
            file_name = input("Please enter the file you would like to load: ")
            if file_name.__contains__(".csv"):
                raw_data_set_full = pd.read_csv("csv_files/" + file_name,
                                                header=4,
                                                skip_blank_lines=True)

                raw_data_set_full.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)

                raw_data_set_full.dropna(how='all', inplace=True)

                raw_data_set_full.fillna(raw_data_set_full.mean(), inplace=True)

                df = raw_data_set_full.values

                incorrect_file = False

        except FileNotFoundError:
            incorrect_file = True
            print("Incorrect file name, please try again...")

    return df


# Start the parse_data method so when this file is run it will prompt the user.
data = parse_data()


# Grab all the columns and separates them into the relevant types of data (location, velocity, overall velocity
# , acceleration, overall acceleration) for each of the markers. Should make it easier to reduce the data set later
# as all I should be using is acceleration and velocity from only two markers.
r_sho_loc = data[:, :3]
r_sho_vel = data[:, 3:6]
r_sho_overVel = data[:, 6:7]
r_sho_acc = data[:, 7:10]
r_sho_overAcc = data[:, 10:11]

arm_si_loc = data[:, 11:14]
arm_si_vel = data[:, 14:17]
arm_si_overVel = data[:, 17:18]
arm_si_acc = data[:, 18:21]
arm_si_overAcc = data[:, 21:22]

r_elb_loc = data[:, 22:25]
r_elb_vel = data[:, 25:28]
r_elb_overVel = data[:, 28:29]
r_elb_acc = data[:, 29:32]
r_elb_overAcc = data[:, 32:33]

r_wri_loc = data[:, 33:36]
r_wri_vel = data[:, 36:39]
r_wri_overVel = data[:, 39:40]
r_wri_acc = data[:, 40:43]
r_wri_overAcc = data[:, 43:44]

l_sho_loc = data[:, 44:47]
l_sho_vel = data[:, 47:50]
l_sho_overVel = data[:, 50:51]
l_sho_acc = data[:, 51:54]
l_sho_overAcc = data[:, 54:55]

l_elb_loc = data[:, 55:58]
l_elb_vel = data[:, 58:61]
l_elb_overVel = data[:, 61:62]
l_elb_acc = data[:, 62:65]
l_elb_overAcc = data[:, 65:66]

l_wri_loc = data[:, 66:69]
l_wri_vel = data[:, 69:72]
l_wri_overVel = data[:, 72:73]
l_wri_acc = data[:, 73:76]
l_wri_overAcc = data[:, 76:77]

chest_loc = data[:, 77:80]
chest_vel = data[:, 80:83]
chest_overVel = data[:, 83:84]
chest_acc = data[:, 84:87]
chest_overAcc = data[:, 87:88]

sob_loc = data[:, 88:91]
sob_vel = data[:, 91:94]
sob_overVel = data[:, 94:95]
sob_acc = data[:, 95:98]
sob_overAcc = data[:, 98:99]

r_hip_loc = data[:, 99:102]
r_hip_vel = data[:, 102:105]
r_hip_overVel = data[:, 105:106]
r_hip_acc = data[:, 106:109]
r_hip_overAcc = data[:, 109:110]

leg_si_loc = data[:, 110:113]
leg_si_vel = data[:, 113:116]
leg_si_overVel = data[:, 116:117]
leg_si_acc = data[:, 117:120]
leg_si_overAcc = data[:, 120:121]

r_kne_loc = data[:, 121:124]
r_kne_vel = data[:, 124:127]
r_kne_overVel = data[:, 127:128]
r_kne_acc = data[:, 128:131]
r_kne_overAcc = data[:, 131:132]

r_ank_loc = data[:, 132:135]
r_ank_vel = data[:, 135:138]
r_ank_overVel = data[:, 138:139]
r_ank_acc = data[:, 139:142]
r_ank_overAcc = data[:, 142:143]

l_hip_loc = data[:, 143:146]
l_hip_vel = data[:, 146:149]
l_hip_overVel = data[:, 149:150]
l_hip_acc = data[:, 150:153]
l_hip_overAcc = data[:, 153:154]

l_kne_loc = data[:, 154:157]
l_kne_vel = data[:, 157:160]
l_kne_overVel = data[:, 160:161]
l_kne_acc = data[:, 161:164]
l_kne_overAcc = data[:, 164:165]

l_ank_loc = data[:, 165:168]
l_ank_vel = data[:, 168:171]
l_ank_overVel = data[:, 171:172]
l_ank_acc = data[:, 172:175]
l_ank_overAcc = data[:, :175:176]

# can be changed to adjust the amount of time I want each frame set to cover (50 covers half a second intervals)
frame_rate = 50

# Lists for sectioning the data into sets of 50 frames (each of these will then be used for feature extraction later)
r_s_l_f_sets = []
r_s_v_f_sets = []
r_s_oV_f_sets = []
r_s_a_f_sets = []
r_s_oA_f_sets = []

a_si_l_f_sets = []
a_si_v_f_sets = []
a_si_oV_f_sets = []
a_si_a_f_sets = []
a_si_oA_f_sets = []

r_e_l_f_sets = []
r_e_v_f_sets = []
r_e_oV_f_sets = []
r_e_a_f_sets = []
r_e_oA_f_sets = []

r_w_l_f_sets = []
r_w_v_f_sets = []
r_w_oV_f_sets = []
r_w_a_f_sets = []
r_w_oA_f_sets = []

l_s_l_f_sets = []
l_s_v_f_sets = []
l_s_oV_f_sets = []
l_s_a_f_sets = []
l_s_oA_f_sets = []

l_e_l_f_sets = []
l_e_v_f_sets = []
l_e_oV_f_sets = []
l_e_a_f_sets = []
l_e_oA_f_sets = []

l_w_l_f_sets = []
l_w_v_f_sets = []
l_w_oV_f_sets = []
l_w_a_f_sets = []
l_w_oA_f_sets = []

c_l_f_sets = []
c_v_f_sets = []
c_oV_f_sets = []
c_a_f_sets = []
c_oA_f_sets = []

s_l_f_sets = []
s_v_f_sets = []
s_oV_f_sets = []
s_a_f_sets = []
s_oA_f_sets = []

r_h_l_f_sets = []
r_h_v_f_sets = []
r_h_oV_f_sets = []
r_h_a_f_sets = []
r_h_oA_f_sets = []

l_si_l_f_sets = []
l_si_v_f_sets = []
l_si_oV_f_sets = []
l_si_a_f_sets = []
l_si_oA_f_sets = []

r_k_l_f_sets = []
r_k_v_f_sets = []
r_k_oV_f_sets = []
r_k_a_f_sets = []
r_k_oA_f_sets = []

r_a_l_f_sets = []
r_a_v_f_sets = []
r_a_oV_f_sets = []
r_a_a_f_sets = []
r_a_oA_f_sets = []

l_h_l_f_sets = []
l_h_v_f_sets = []
l_h_oV_f_sets = []
l_h_a_f_sets = []
l_h_oA_f_sets = []

l_k_l_f_sets = []
l_k_v_f_sets = []
l_k_oV_f_sets = []
l_k_a_f_sets = []
l_k_oA_f_sets = []

l_a_l_f_sets = []
l_a_v_f_sets = []
l_a_oV_f_sets = []
l_a_a_f_sets = []
l_a_oA_f_sets = []

# Grab the data and put it into lists based upon the specified frame_rate chosen. Currently sets of 50 frames
for i in range(0, (r_sho_loc.shape[0] - 50), frame_rate):
    r_s_l_f_sets.append(r_sho_loc[i:i + frame_rate, :])
    r_s_v_f_sets.append(r_sho_vel[i:i + frame_rate, :])
    r_s_oV_f_sets.append(r_sho_overVel[i:i + frame_rate, :])
    r_s_a_f_sets.append(r_sho_acc[i:i + frame_rate, :])
    r_s_oA_f_sets.append(r_sho_overAcc[i:i + frame_rate, :])

    a_si_l_f_sets.append(arm_si_loc[i:i + frame_rate, :])
    a_si_v_f_sets.append(arm_si_vel[i:i + frame_rate, :])
    a_si_oV_f_sets.append(arm_si_overVel[i:i + frame_rate, :])
    a_si_a_f_sets.append(arm_si_acc[i:i + frame_rate, :])
    a_si_oA_f_sets.append(arm_si_overAcc[i:i + frame_rate, :])

    r_e_l_f_sets.append(r_elb_loc[i:i + frame_rate, :])
    r_e_v_f_sets.append(r_elb_vel[i:i + frame_rate, :])
    r_e_oV_f_sets.append(r_elb_overVel[i:i + frame_rate, :])
    r_e_a_f_sets.append(r_elb_acc[i:i + frame_rate, :])
    r_e_oA_f_sets.append(r_elb_overAcc[i:i + frame_rate, :])

    r_w_l_f_sets.append(r_wri_loc[i:i + frame_rate, :])
    r_w_v_f_sets.append(r_wri_vel[i:i + frame_rate, :])
    r_w_oV_f_sets.append(r_wri_overVel[i:i + frame_rate, :])
    r_w_a_f_sets.append(r_wri_acc[i:i + frame_rate, :])
    r_w_oA_f_sets.append(r_wri_overAcc[i:i + frame_rate, :])

    l_s_l_f_sets.append(l_sho_loc[i:i + frame_rate, :])
    l_s_v_f_sets.append(l_sho_vel[i:i + frame_rate, :])
    l_s_oV_f_sets.append(l_sho_overVel[i:i + frame_rate, :])
    l_s_a_f_sets.append(l_sho_acc[i:i + frame_rate, :])
    l_s_oA_f_sets.append(l_sho_overAcc[i:i + frame_rate, :])

    l_e_l_f_sets.append(l_elb_loc[i:i + frame_rate, :])
    l_e_v_f_sets.append(l_elb_vel[i:i + frame_rate, :])
    l_e_oV_f_sets.append(l_elb_overVel[i:i + frame_rate, :])
    l_e_a_f_sets.append(l_elb_acc[i:i + frame_rate, :])
    l_e_oA_f_sets.append(l_elb_overAcc[i:i + frame_rate, :])

    l_w_l_f_sets.append(l_wri_loc[i:i + frame_rate, :])
    l_w_v_f_sets.append(l_wri_vel[i:i + frame_rate, :])
    l_w_oV_f_sets.append(l_wri_overVel[i:i + frame_rate, :])
    l_w_a_f_sets.append(l_wri_acc[i:i + frame_rate, :])
    l_w_oA_f_sets.append(l_wri_overAcc[i:i + frame_rate, :])

    c_l_f_sets.append(chest_loc[i:i + frame_rate, :])
    c_v_f_sets.append(chest_vel[i:i + frame_rate, :])
    c_oV_f_sets.append(chest_overVel[i:i + frame_rate, :])
    c_a_f_sets.append(chest_acc[i:i + frame_rate, :])
    c_oA_f_sets.append(chest_overAcc[i:i + frame_rate, :])

    s_l_f_sets.append(sob_loc[i:i + frame_rate, :])
    s_v_f_sets.append(sob_vel[i:i + frame_rate, :])
    s_oV_f_sets.append(sob_overVel[i:i + frame_rate, :])
    s_a_f_sets.append(sob_acc[i:i + frame_rate, :])
    s_oA_f_sets.append(sob_overAcc[i:i + frame_rate, :])

    r_h_l_f_sets.append(r_hip_loc[i:i + frame_rate, :])
    r_h_v_f_sets.append(r_hip_vel[i:i + frame_rate, :])
    r_h_oV_f_sets.append(r_hip_overVel[i:i + frame_rate, :])
    r_h_a_f_sets.append(r_hip_acc[i:i + frame_rate, :])
    r_h_oA_f_sets.append(r_hip_overAcc[i:i + frame_rate, :])

    l_si_l_f_sets.append(leg_si_loc[i:i + frame_rate, :])
    l_si_v_f_sets.append(leg_si_vel[i:i + frame_rate, :])
    l_si_oV_f_sets.append(leg_si_overVel[i:i + frame_rate, :])
    l_si_a_f_sets.append(leg_si_acc[i:i + frame_rate, :])
    l_si_oA_f_sets.append(leg_si_overAcc[i:i + frame_rate, :])

    r_k_l_f_sets.append(r_kne_loc[i:i + frame_rate, :])
    r_k_v_f_sets.append(r_kne_vel[i:i + frame_rate, :])
    r_k_oV_f_sets.append(r_kne_overVel[i:i + frame_rate, :])
    r_k_a_f_sets.append(r_kne_acc[i:i + frame_rate, :])
    r_k_oA_f_sets.append(r_kne_overAcc[i:i + frame_rate, :])

    r_a_l_f_sets.append(r_ank_loc[i:i + frame_rate, :])
    r_a_v_f_sets.append(r_ank_vel[i:i + frame_rate, :])
    r_a_oV_f_sets.append(r_ank_overVel[i:i + frame_rate, :])
    r_a_a_f_sets.append(r_ank_acc[i:i + frame_rate, :])
    r_a_oA_f_sets.append(r_ank_overAcc[i:i + frame_rate, :])

    l_h_l_f_sets.append(l_hip_loc[i:i + frame_rate, :])
    l_h_v_f_sets.append(l_hip_vel[i:i + frame_rate, :])
    l_h_oV_f_sets.append(l_hip_overVel[i:i + frame_rate, :])
    l_h_a_f_sets.append(l_hip_acc[i:i + frame_rate, :])
    l_h_oA_f_sets.append(l_hip_overAcc[i:i + frame_rate, :])

    l_k_l_f_sets.append(l_kne_loc[i:i + frame_rate, :])
    l_k_v_f_sets.append(l_kne_vel[i:i + frame_rate, :])
    l_k_oV_f_sets.append(l_kne_overVel[i:i + frame_rate, :])
    l_k_a_f_sets.append(l_kne_acc[i:i + frame_rate, :])
    l_k_oA_f_sets.append(l_kne_overAcc[i:i + frame_rate, :])

    l_a_l_f_sets.append(l_ank_loc[i:i + frame_rate, :])
    l_a_v_f_sets.append(l_ank_vel[i:i + frame_rate, :])
    l_a_oV_f_sets.append(l_ank_overVel[i:i + frame_rate, :])
    l_a_a_f_sets.append(l_ank_acc[i:i + frame_rate, :])
    l_a_oA_f_sets.append(l_ank_overAcc[i:i + frame_rate, :])

right_shoulder = [r_s_l_f_sets, r_s_v_f_sets, r_s_a_f_sets]
arm_side_identifier = [a_si_l_f_sets, a_si_v_f_sets, a_si_a_f_sets]
right_elbow = [r_e_l_f_sets, r_e_v_f_sets, r_e_a_f_sets]
right_wrist = [r_w_l_f_sets, r_w_v_f_sets, r_w_a_f_sets]
left_shoulder = [l_s_l_f_sets, l_s_v_f_sets, l_s_a_f_sets]
left_elbow = [l_e_l_f_sets, l_e_v_f_sets, l_e_a_f_sets]
left_wrist = [l_w_l_f_sets, l_w_v_f_sets, l_w_a_f_sets]
chest = [c_l_f_sets, c_v_f_sets, c_a_f_sets]
small_of_back = [s_l_f_sets, s_v_f_sets, s_a_f_sets]
right_hip = [r_h_l_f_sets, r_h_v_f_sets, r_h_a_f_sets]
leg_side_identifier = [l_si_l_f_sets, l_si_v_f_sets, l_si_a_f_sets]
right_knee = [r_k_l_f_sets, r_k_v_f_sets, r_k_a_f_sets]
right_ankle = [r_a_l_f_sets, r_a_v_f_sets, r_a_a_f_sets]
left_hip = [l_h_l_f_sets, l_h_v_f_sets, l_h_a_f_sets]
left_knee = [l_k_l_f_sets, l_k_v_f_sets, l_k_a_f_sets]
left_ankle = [l_a_l_f_sets, l_a_v_f_sets, l_a_a_f_sets]

data_for_feature_extraction = [right_shoulder, arm_side_identifier, right_elbow, right_wrist,
                               left_shoulder, left_elbow, left_wrist,
                               chest, small_of_back,
                               right_hip, leg_side_identifier, right_knee, right_ankle,
                               left_hip, left_knee, left_ankle
                               ]

reduced_data = [left_wrist[2], leg_side_identifier[2]]

print(*reduced_data[0], sep='\n')

# Saving the data into a file to avoid having to do this every single time I want to test on the data.
with open("pickle_files/walkingData", 'wb') as f:
    pick.dump(data_for_feature_extraction, f)
    f.close()

with open("pickle_files/redWalkingData", 'wb') as f:
    pick.dump(reduced_data, f)
    f.close()



# Check this is equal to the frame_rate value to ensure the sets are of the correct size.
print(len(data_for_feature_extraction[0][0]))
