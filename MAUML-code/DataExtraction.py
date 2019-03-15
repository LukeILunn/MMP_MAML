import pandas as pd
import statistics as stat


def parse_data():
    df = pd.DataFrame
    incorrect_file = True
    while incorrect_file:
        try:
            file_name = input("Please enter the file you would like to load: ")
            raw_data_set_full = pd.read_csv("csv_files/" + file_name,
                                            header=4,
                                            skip_blank_lines=True)

            raw_data_set_full.drop(['Unnamed: 0', 'Unnamed: 1'], axis=1, inplace=True)

            raw_data_set_full.dropna(how='all', inplace=True)

            df = raw_data_set_full.values

        except FileNotFoundError:
            incorrect_file = True
            print("Incorrect file name, please try again...")
        else:
            incorrect_file = False

    return df


def feature_extract(this_list):
    minim = min(this_list)
    maxim = max(this_list)
    average = stat.mean(this_list)
    stdevi = stat.stdev(this_list)
    feature_set = [minim, maxim, average, stdevi]
    return feature_set


data = parse_data()

r_sho = data[:, :11]
arm_si = data[:, 11:22]
r_elb = data[:, 22:33]
r_wri = data[:, 33:44]
l_sho = data[:, 44:55]
l_elb = data[:, 55:66]
l_wri = data[:, 66:77]
chest = data[:, 77:88]
sob = data[:, 88:99]
r_hip = data[:, 99:110]
leg_si = data[:, 110:121]
r_kne = data[:, 121:132]
r_ank = data[:, 132:143]
l_hip = data[:, 143:154]
l_kne = data[:, 154:165]
l_ank = data[:, 165:176]

data_separated = [r_sho, arm_si, r_elb, r_wri,
                  l_sho, l_elb, l_wri,
                  chest, sob,
                  r_hip, leg_si, r_kne, r_ank,
                  l_hip, l_kne, l_ank]

# for i in range(len(data_separated)):
#     print("\n\n")
#     print(data_separated[i][0])

print(data)
