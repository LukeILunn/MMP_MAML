import pandas as pd
import pickle as pick
import numpy as np

# These variables are used across different methods and so may need to be called using global. The first two are used
# to decide whether to continue looping and asking the user for input and store the new file name if necessary.
# "time_step" is used to decide the amount of frames to be stored in a single list when they are appended.
f_name = ""
more_input = True
time_step = 200


# Method to grab the data from a given csv file (input will only find files in the csv_files folder and only
# files containing the file type .csv), drops the necessary columns and returns the raw data as a pandas data frame
def extract_data():
    global f_name
    df = pd.DataFrame
    incorrect_file = True
    while incorrect_file:
        try:
            f_name = input("Please enter the file you would like to load: ")
            if f_name.__contains__(".csv"):
                raw_data_set_full = pd.read_csv("csv_files/" + f_name,
                                                header=4,
                                                skip_blank_lines=True)

                raw_data_set_full.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)

                raw_data_set_full.dropna(how='all', inplace=True)

                #raw_data_set_full.fillna(raw_data_set_full.mean(), inplace=True)

                df = raw_data_set_full.values

                incorrect_file = False

        except FileNotFoundError:
            incorrect_file = True
            print("Incorrect file name, please try again...")

    return df


def fill_NaNs(from_list):
    ret = from_list[i:i + time_step, :]
    mean_of_columns = np.nanmean(ret, axis=0)
    nans = np.where(np.isnan(ret))
    ret[nans] = np.take(mean_of_columns, nans[1])
    return ret


while more_input:
    # Start the extract_data method so when this file is run it will prompt the user.
    data = extract_data()


    # Grab all the columns and separates them into the relevant types of data (location, velocity, overall velocity
    # , acceleration, overall acceleration) for each of the markers. Should make it easier to reduce the data set later
    # as all I should be using is acceleration and velocity from only two markers.
    r_sho_loc = data[:, :3]
    r_sho_vel = data[:, 3:6]
    r_sho_acc = data[:, 7:10]

    arm_si_loc = data[:, 11:14]
    arm_si_vel = data[:, 14:17]
    arm_si_acc = data[:, 18:21]

    r_elb_loc = data[:, 22:25]
    r_elb_vel = data[:, 25:28]
    r_elb_acc = data[:, 29:32]

    r_wri_loc = data[:, 33:36]
    r_wri_vel = data[:, 36:39]
    r_wri_acc = data[:, 40:43]

    l_sho_loc = data[:, 44:47]
    l_sho_vel = data[:, 47:50]
    l_sho_acc = data[:, 51:54]

    l_elb_loc = data[:, 55:58]
    l_elb_vel = data[:, 58:61]
    l_elb_acc = data[:, 62:65]

    l_wri_loc = data[:, 66:69]
    l_wri_vel = data[:, 69:72]
    l_wri_acc = data[:, 73:76]

    chest_loc = data[:, 77:80]
    chest_vel = data[:, 80:83]
    chest_acc = data[:, 84:87]

    sob_loc = data[:, 88:91]
    sob_vel = data[:, 91:94]
    sob_acc = data[:, 95:98]

    r_hip_loc = data[:, 99:102]
    r_hip_vel = data[:, 102:105]
    r_hip_acc = data[:, 106:109]

    leg_si_loc = data[:, 110:113]
    leg_si_vel = data[:, 113:116]
    leg_si_acc = data[:, 117:120]

    r_kne_loc = data[:, 121:124]
    r_kne_vel = data[:, 124:127]
    r_kne_acc = data[:, 128:131]

    r_ank_loc = data[:, 132:135]
    r_ank_vel = data[:, 135:138]
    r_ank_acc = data[:, 139:142]

    l_hip_loc = data[:, 143:146]
    l_hip_vel = data[:, 146:149]
    l_hip_acc = data[:, 150:153]

    l_kne_loc = data[:, 154:157]
    l_kne_vel = data[:, 157:160]
    l_kne_acc = data[:, 161:164]

    l_ank_loc = data[:, 165:168]
    l_ank_vel = data[:, 168:171]
    l_ank_acc = data[:, 172:175]

    # Lists for sectioning the data into sets of frames (each of these will then be used for feature extraction later)
    r_s_l_f_sets = []
    r_s_v_f_sets = []
    r_s_a_f_sets = []

    a_si_l_f_sets = []
    a_si_v_f_sets = []
    a_si_a_f_sets = []

    r_e_l_f_sets = []
    r_e_v_f_sets = []
    r_e_a_f_sets = []

    r_w_l_f_sets = []
    r_w_v_f_sets = []
    r_w_a_f_sets = []

    l_s_l_f_sets = []
    l_s_v_f_sets = []
    l_s_a_f_sets = []

    l_e_l_f_sets = []
    l_e_v_f_sets = []
    l_e_a_f_sets = []

    l_w_l_f_sets = []
    l_w_v_f_sets = []
    l_w_a_f_sets = []

    c_l_f_sets = []
    c_v_f_sets = []
    c_a_f_sets = []

    s_l_f_sets = []
    s_v_f_sets = []
    s_a_f_sets = []

    r_h_l_f_sets = []
    r_h_v_f_sets = []
    r_h_a_f_sets = []

    l_si_l_f_sets = []
    l_si_v_f_sets = []
    l_si_a_f_sets = []

    r_k_l_f_sets = []
    r_k_v_f_sets = []
    r_k_a_f_sets = []

    r_a_l_f_sets = []
    r_a_v_f_sets = []
    r_a_a_f_sets = []

    l_h_l_f_sets = []
    l_h_v_f_sets = []
    l_h_a_f_sets = []

    l_k_l_f_sets = []
    l_k_v_f_sets = []
    l_k_a_f_sets = []

    l_a_l_f_sets = []
    l_a_v_f_sets = []
    l_a_a_f_sets = []

    # Grab the data and put it into lists based upon the specified time_step chosen. Currently sets of 50 frames
    for i in range(0, (r_sho_loc.shape[0] - time_step), time_step):

        loc_filler = fill_NaNs(r_sho_loc)
        vel_filler = fill_NaNs(r_sho_vel)
        acc_filler = fill_NaNs(r_sho_acc)
        r_s_l_f_sets.append(loc_filler)
        r_s_v_f_sets.append(vel_filler)
        r_s_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(arm_si_loc)
        vel_filler = fill_NaNs(arm_si_vel)
        acc_filler = fill_NaNs(arm_si_acc)
        a_si_l_f_sets.append(loc_filler)
        a_si_v_f_sets.append(vel_filler)
        a_si_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(r_elb_loc)
        vel_filler = fill_NaNs(r_elb_vel)
        acc_filler = fill_NaNs(r_elb_acc)
        r_e_l_f_sets.append(loc_filler)
        r_e_v_f_sets.append(vel_filler)
        r_e_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(r_wri_loc)
        vel_filler = fill_NaNs(r_wri_vel)
        acc_filler = fill_NaNs(r_wri_acc)
        r_w_l_f_sets.append(loc_filler)
        r_w_v_f_sets.append(vel_filler)
        r_w_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_sho_loc)
        vel_filler = fill_NaNs(l_sho_vel)
        acc_filler = fill_NaNs(l_sho_acc)
        l_s_l_f_sets.append(loc_filler)
        l_s_v_f_sets.append(vel_filler)
        l_s_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_elb_loc)
        vel_filler = fill_NaNs(l_elb_vel)
        acc_filler = fill_NaNs(l_elb_acc)
        l_e_l_f_sets.append(loc_filler)
        l_e_v_f_sets.append(vel_filler)
        l_e_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_wri_loc)
        vel_filler = fill_NaNs(l_wri_vel)
        acc_filler = fill_NaNs(l_wri_acc)
        l_w_l_f_sets.append(loc_filler)
        l_w_v_f_sets.append(vel_filler)
        l_w_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(chest_loc)
        vel_filler = fill_NaNs(chest_vel)
        acc_filler = fill_NaNs(chest_acc)
        c_l_f_sets.append(loc_filler)
        c_v_f_sets.append(vel_filler)
        c_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(sob_loc)
        vel_filler = fill_NaNs(sob_vel)
        acc_filler = fill_NaNs(sob_acc)
        s_l_f_sets.append(loc_filler)
        s_v_f_sets.append(vel_filler)
        s_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(r_hip_loc)
        vel_filler = fill_NaNs(r_hip_vel)
        acc_filler = fill_NaNs(r_hip_acc)
        r_h_l_f_sets.append(loc_filler)
        r_h_v_f_sets.append(vel_filler)
        r_h_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(leg_si_loc)
        vel_filler = fill_NaNs(leg_si_vel)
        acc_filler = fill_NaNs(leg_si_acc)
        l_si_l_f_sets.append(loc_filler)
        l_si_v_f_sets.append(vel_filler)
        l_si_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(r_kne_loc)
        vel_filler = fill_NaNs(r_kne_vel)
        acc_filler = fill_NaNs(r_kne_acc)
        r_k_l_f_sets.append(loc_filler)
        r_k_v_f_sets.append(vel_filler)
        r_k_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(r_ank_loc)
        vel_filler = fill_NaNs(r_ank_vel)
        acc_filler = fill_NaNs(r_ank_acc)
        r_a_l_f_sets.append(loc_filler)
        r_a_v_f_sets.append(vel_filler)
        r_a_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_hip_loc)
        vel_filler = fill_NaNs(l_hip_vel)
        acc_filler = fill_NaNs(l_hip_acc)
        l_h_l_f_sets.append(loc_filler)
        l_h_v_f_sets.append(vel_filler)
        l_h_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_kne_loc)
        vel_filler = fill_NaNs(l_kne_vel)
        acc_filler = fill_NaNs(l_kne_acc)
        l_k_l_f_sets.append(loc_filler)
        l_k_v_f_sets.append(vel_filler)
        l_k_a_f_sets.append(acc_filler)

        loc_filler = fill_NaNs(l_ank_loc)
        vel_filler = fill_NaNs(l_ank_vel)
        acc_filler = fill_NaNs(l_ank_acc)
        l_a_l_f_sets.append(loc_filler)
        l_a_v_f_sets.append(vel_filler)
        l_a_a_f_sets.append(acc_filler)

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

    reduced_data = [right_wrist[2], leg_side_identifier[2]]

    # Saving the data into a file to avoid having to do this every single time I want to test on the data.
    if f_name.__contains__("walking"):
        with open("pickle_files/DataExtraction_results/walkingData", 'wb') as f:
            pick.dump(data_for_feature_extraction, f)
            f.close()

        with open("pickle_files/DataExtraction_results/redWalkingData", 'wb') as f:
            pick.dump(reduced_data, f)
            f.close()
    elif f_name.__contains__("sat"):
        with open("pickle_files/DataExtraction_results/satData", 'wb') as f:
            pick.dump(data_for_feature_extraction, f)
            f.close()

        with open("pickle_files/DataExtraction_results/redSittingData", 'wb') as f:
            pick.dump(reduced_data, f)
            f.close()
    elif f_name.__contains__("stood"):
        with open("pickle_files/DataExtraction_results/stoodData", 'wb') as f:
            pick.dump(data_for_feature_extraction, f)
            f.close()

        with open("pickle_files/DataExtraction_results/redStandingData", 'wb') as f:
            pick.dump(reduced_data, f)
            f.close()


    # Check this is equal to the frame_rate value to ensure the sets are of the correct size.
    print(len(data_for_feature_extraction[0][0]))
    reply = input("\n\n\nWould you like to enter another file? [y] [n]")
    reply = reply.lower()
    if reply.__contains__("y"):
        more_input = True
    elif reply.__contains__("n"):
        more_input = False

print("\n\nData extraction complete, files should exist in pickle_files directory.")