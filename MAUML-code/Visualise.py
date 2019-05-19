import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The creation of the array which is used as the baseline indicator.
chance = np.array([0.5] * 30)


# This method will be run as a default for a complete run of all of the scripts. This will produce a comparison
# Graph with the optimal settings, as discovered from the tests performed.
def visualise_full_vs_red_vs_raw():
    with open("pickle_files/final_comparison/cross_val_avgs", 'rb') as f:
        cross_val_avgs = pickle.load(f)
        f.close()

    with open("pickle_files/final_comparison/cross_val_stdev", 'rb') as f:
        cross_val_stdev = pickle.load(f)
        f.close()

    # Empty NumPy arrays for storing the values which were loaded using pickle
    full_avgs = np.array([])
    red_avgs = np.array([])
    raw_avgs = np.array([])

    full_stdev = np.array([])
    red_stdev = np.array([])
    raw_stdev = np.array([])

    # Append all of the average and standard deviation values to their relavant variables
    for i in range(len(cross_val_avgs)):
        full_avgs = np.append(cross_val_avgs[i][0], full_avgs)
        red_avgs = np.append(cross_val_avgs[i][1], red_avgs)
        raw_avgs = np.append(cross_val_avgs[i][2], raw_avgs)
        full_stdev = np.append(cross_val_stdev[i][0], full_stdev)
        red_stdev = np.append(cross_val_stdev[i][1], red_stdev)
        raw_stdev = np.append(cross_val_stdev[i][2], raw_stdev)

    # Store the average standard deviation for use in ax.fill_between() which creates the error band
    # in the plot.
    full_avg_err = full_stdev.mean()
    red_avg_err = red_stdev.mean()
    raw_avg_err = raw_stdev.mean()

    # The data to be plotted is stored in this DataFrame for easy plotting by calling the relevant y label.
    df = pd.DataFrame({'x': range(1, 31), 'y1': full_avgs, 'y2': red_avgs, 'y3': raw_avgs, 'y4': chance})

    # The range here is used for the error band plotting as the DataFrame cannot be used in the fill_between()
    # method.
    x = range(1, 31)

    # Create the necessary objects for plotting.
    fig = plt.figure()
    ax = plt.subplot(111)

    # Plot all the lines and the error bands if left uncommented. To remove error bands and just plot lines
    # comment lines 60, 62, and 64.
    ax.plot('x', 'y4', data=df, marker='', color='darkorange', linewidth=3, label='baseline')
    ax.plot('x', 'y1', data=df, marker='', color='black', linewidth=1.5, label='full markers with features')
    ax.fill_between(x, full_avgs - full_avg_err, full_avgs + full_avg_err)
    ax.plot('x', 'y2', data=df, marker='', color='olive', linewidth=1.5, label='reduced set with features')
    ax.fill_between(x, red_avgs - red_avg_err, red_avgs + red_avg_err)
    ax.plot('x', 'y3', data=df, marker='', color='red', linewidth=1.5, label='reduced set raw data')
    ax.fill_between(x, raw_avgs - raw_avg_err, raw_avgs + raw_avg_err)

    # Set location of the legend within the plot, sometimes may also need n_cols set if there are many lines.
    ax.legend(loc=3)

    # Setting the numbers upon the x-axis and y-axis and the values between each tick.
    plt.xticks(np.arange(0, 30.1, 2))
    plt.yticks(np.arange(0.00, 1.01, 0.05))

    #Creating a grid to make it clearer where the lines actually sit on the axes.
    ax.grid(color='black', linestyle='-', linewidth=0.2)

    # Setting various labels and limits.
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Average Accuracy (%)')
    plt.title("K-fold Cross Validation Scores (Time step: 200, reduced set: rw and lsi)\n"
              "Optimal setting after tests")
    plt.ylim(0.00, 1.00)

    # Once all settings are appropriate the plot can be shown and shoul appear
    plt.show()

    # The overall averages are stored here to be printed, these can provide some clarity and augment the
    # plots well.
    full_overall_avg = np.mean(full_avgs)
    red_overall_avg = np.mean(red_avgs)
    raw_overall_avg = np.mean(raw_avgs)

    print("The overall average accuracy for the full marker set with features was: ",
          full_overall_avg)
    print("The overall average accuracy for the reduced marker set with features was: ",
          red_overall_avg)
    print("The overall average accuracy for the reduced marker set with raw data was: ", raw_overall_avg)
    print("\n\n")
    print("The average standard deviation for full: ", full_avg_err)
    print("for reduced with features: ", red_avg_err)
    print("for reduced with raw data: ", raw_avg_err)


# This method was used to visualise the tests for different marker combinations. It works in very much the same way
# as the visualise_full_vs_red_vs_raw() method, but with more variables to be plotted.
def visualise_red_marker_combinations():
    with open("pickle_files/reduced_marker_tests/cross_val_stdev_rw_rh", 'rb') as f:
        rw_rh_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_rw_lh", 'rb') as f:
        rw_lh_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_rw_lsi", 'rb') as f:
        rw_lsi_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_rw_sob", 'rb') as f:
        rw_sob_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_lw_rh", 'rb') as f:
        lw_rh_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_lw_lh", 'rb') as f:
        lw_lh_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_lw_lsi", 'rb') as f:
        lw_lsi_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_stdev_lw_sob", 'rb') as f:
        lw_sob_std = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_rw_rh", 'rb') as f:
        rw_rh_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_rw_lh", 'rb') as f:
        rw_lh_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_rw_lsi", 'rb') as f:
        rw_lsi_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_rw_sob", 'rb') as f:
        rw_sob_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_lw_rh", 'rb') as f:
        lw_rh_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_lw_lh", 'rb') as f:
        lw_lh_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_lw_lsi", 'rb') as f:
        lw_lsi_avg = pickle.load(f)
        f.close()

    with open("pickle_files/reduced_marker_tests/cross_val_avgs_lw_sob", 'rb') as f:
        lw_sob_avg = pickle.load(f)
        f.close()

    rw_rh_avg = np.asarray(rw_rh_avg)
    rw_rh_avg = rw_rh_avg.flatten()
    rw_lh_avg = np.asarray(rw_lh_avg)
    rw_lh_avg = rw_lh_avg.flatten()
    rw_lsi_avg = np.asarray(rw_lsi_avg)
    rw_lsi_avg = rw_lsi_avg.flatten()
    rw_sob_avg = np.asarray(rw_sob_avg)
    rw_sob_avg = rw_sob_avg.flatten()
    lw_rh_avg = np.asarray(lw_rh_avg)
    lw_rh_avg = lw_rh_avg.flatten()
    lw_lh_avg = np.asarray(lw_lh_avg)
    lw_lh_avg = lw_lh_avg.flatten()
    lw_lsi_avg = np.asarray(lw_lsi_avg)
    lw_lsi_avg = lw_lsi_avg.flatten()
    lw_sob_avg = np.asarray(lw_sob_avg)
    lw_sob_avg = lw_sob_avg.flatten()

    rw_rh_std = np.asarray(rw_rh_std)
    rw_rh_std = rw_rh_std.flatten()
    rw_lh_std = np.asarray(rw_lh_std)
    rw_lh_std = rw_lh_std.flatten()
    rw_lsi_std = np.asarray(rw_lsi_std)
    rw_lsi_std = rw_lsi_std.flatten()
    rw_sob_std = np.asarray(rw_sob_std)
    rw_sob_std = rw_sob_std.flatten()
    lw_rh_std = np.asarray(lw_rh_std)
    lw_rh_std = lw_rh_std.flatten()
    lw_lh_std = np.asarray(lw_lh_std)
    lw_lh_std = lw_lh_std.flatten()
    lw_lsi_std = np.asarray(lw_lsi_std)
    lw_lsi_std = lw_lsi_std.flatten()
    lw_sob_std = np.asarray(lw_sob_std)
    lw_sob_std = lw_sob_std.flatten()

    df = pd.DataFrame({'x': range(0, 30), 'y1': rw_rh_avg, 'y2': rw_lh_avg, 'y3': rw_lsi_avg, 'y4': rw_sob_avg,
                       'y5': lw_rh_avg, 'y6': lw_lh_avg, 'y7': lw_lsi_avg, 'y8': lw_sob_avg, 'y9': chance})

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot('x', 'y9', data=df, marker='', color='black', linewidth=3, label='baseline')
    ax.plot('x', 'y1', data=df, marker='', color='olive', linewidth=1.5, label='rw_rh')
    ax.plot('x', 'y2', data=df, marker='', color='blue', linewidth=1.5, label='rw_lh')
    ax.plot('x', 'y3', data=df, marker='', color='orange', linewidth=1.5, label='rw_lsi')
    ax.plot('x', 'y4', data=df, marker='', color='purple', linewidth=1.5, label='rw_sob')
    ax.plot('x', 'y5', data=df, marker='', color='red', linewidth=1.5, label='lw_rh')
    ax.plot('x', 'y6', data=df, marker='', color='limegreen', linewidth=1.5, label='lw_lh')
    ax.plot('x', 'y7', data=df, marker='', color='yellow', linewidth=1.5, label='lw_lsi')
    ax.plot('x', 'y8', data=df, marker='', color='pink', linewidth=1.5, label='lw_sob')
    plt.xticks(np.arange(0, 30, 2))
    plt.yticks(np.arange(0.00, 1.01, 0.05))
    ax.legend(loc=3)
    ax.grid(color='black', linestyle='-', linewidth=0.2)
    ax.set_xlabel('Test Number')
    ax.set_ylabel('Average Accuracy (%)')
    plt.title("K-fold Cross Validation Scores\n "
              "Time step: 100, reduced set: testing different marker combinations")
    plt.ylim(0.00, 1.00)
    plt.show()

    print("The overall average accuracy for right wrist with right hip was: ",
          rw_rh_avg.mean())
    print("The overall average accuracy for right wrist with left hip was: ",
          rw_lh_avg.mean())
    print("The overall average accuracy for right wrist with leg si was: ",
          rw_lsi_avg.mean())
    print("The overall average accuracy for right wrist with small of back was: ",
          rw_sob_avg.mean())
    print("The overall average accuracy for left wrist with right hip was: ",
          lw_rh_avg.mean())
    print("The overall average accuracy for left wrist with left hip was: ",
          lw_lh_avg.mean())
    print("The overall average accuracy for left wrist with leg si was: ",
          lw_lsi_avg.mean())
    print("The overall average accuracy for left wrist with small of back was: ",
          lw_sob_avg.mean())
    print("\n\n")
    print("The average standard deviation for right wrist with right hip was: ", rw_rh_std.mean())
    print("for right wrist with left hip: ", rw_lh_std.mean())
    print("for right wrist with leg si: ", rw_lsi_std.mean())
    print("for right wrist with small of back: ", rw_sob_std.mean())
    print("for left wrist with right hip: ", lw_rh_std.mean())
    print("for left wrist with left hip: ", lw_lh_std.mean())
    print("for left wrist with leg si: ", lw_lsi_std.mean())
    print("for left wrist with small of back: ", lw_sob_std.mean())


# Visualising the time step comparison became slightly different where I began to use looping to load the
# variables. The code is again very much the same until there is a different plot at the bottom of the method
# by providing a 0, 1, or 2 the method will create 1 of 3 types of plot. 0 shows all of the time step lines
# on the same plot with the 30 tests as the x-axis. 1 will show a progression of average accuracy with increasing
# time step values, with time step value as the x-axis. 2 will show the same as 1 but for standard deviation, which
# becomes the y-axis.
def visualise_time_step_comparsion(which_plot):
    time_steps = ["25", "50", "75", "100", "125", "150", "175", "200", "225", "250", "275", "300"]
    time_step_avgs = []
    time_step_stds = []

    for i in range(len(time_steps)):
        f_name = "pickle_files/time_step_tests/cross_val_avgs_" + time_steps[i]
        with open(f_name, 'rb') as f:
            var = pickle.load(f)
            time_step_avgs.append(var)
            f.close()

        f_name = "pickle_files/time_step_tests/cross_val_stdev_" + time_steps[i]
        with open(f_name, 'rb') as f:
            var = pickle.load(f)
            time_step_stds.append(var)
            f.close()

    step_25 = time_step_avgs[0]
    step_25 = np.asarray(step_25)
    step_25 = step_25.flatten()
    stds_25 = time_step_stds[0]
    stds_25 = np.asarray(stds_25)
    stds_25 = stds_25.flatten()

    step_50 = time_step_avgs[1]
    step_50 = np.asarray(step_50)
    step_50 = step_50.flatten()
    stds_50 = time_step_stds[1]
    stds_50 = np.asarray(stds_50)
    stds_50 = stds_50.flatten()

    step_75 = time_step_avgs[2]
    step_75 = np.asarray(step_75)
    step_75 = step_75.flatten()
    stds_75 = time_step_stds[2]
    stds_75 = np.asarray(stds_75)
    stds_75 = stds_75.flatten()

    step_100 = time_step_avgs[3]
    step_100 = np.asarray(step_100)
    step_100 = step_100.flatten()
    stds_100 = time_step_stds[3]
    stds_100 = np.asarray(stds_100)
    stds_100 = stds_100.flatten()

    step_125 = time_step_avgs[4]
    step_125 = np.asarray(step_125)
    step_125 = step_125.flatten()
    stds_125 = time_step_stds[4]
    stds_125 = np.asarray(stds_125)
    stds_125 = stds_125.flatten()

    step_150 = time_step_avgs[5]
    step_150 = np.asarray(step_150)
    step_150 = step_150.flatten()
    stds_150 = time_step_stds[5]
    stds_150 = np.asarray(stds_150)
    stds_150 = stds_150.flatten()

    step_175 = time_step_avgs[6]
    step_175 = np.asarray(step_175)
    step_175 = step_175.flatten()
    stds_175 = time_step_stds[6]
    stds_175 = np.asarray(stds_175)
    stds_175 = stds_175.flatten()

    step_200 = time_step_avgs[7]
    step_200 = np.asarray(step_200)
    step_200 = step_200.flatten()
    stds_200 = time_step_stds[7]
    stds_200 = np.asarray(stds_200)
    stds_200 = stds_200.flatten()

    step_225 = time_step_avgs[8]
    step_225 = np.asarray(step_225)
    step_225 = step_225.flatten()
    stds_225 = time_step_stds[8]
    stds_225 = np.asarray(stds_225)
    stds_225 = stds_225.flatten()

    step_250 = time_step_avgs[9]
    step_250 = np.asarray(step_250)
    step_250 = step_250.flatten()
    stds_250 = time_step_stds[9]
    stds_250 = np.asarray(stds_250)
    stds_250 = stds_250.flatten()

    step_275 = time_step_avgs[10]
    step_275 = np.asarray(step_275)
    step_275 = step_275.flatten()
    stds_275 = time_step_stds[10]
    stds_275 = np.asarray(stds_275)
    stds_275 = stds_275.flatten()

    step_300 = time_step_avgs[11]
    step_300 = np.asarray(step_300)
    step_300 = step_300.flatten()
    stds_300 = time_step_stds[11]
    stds_300 = np.asarray(stds_300)
    stds_300 = stds_300.flatten()

    overall_avgs = np.array([])
    overall_avgs = np.append(step_300.mean(), overall_avgs)
    overall_avgs = np.append(step_275.mean(), overall_avgs)
    overall_avgs = np.append(step_250.mean(), overall_avgs)
    overall_avgs = np.append(step_225.mean(), overall_avgs)
    overall_avgs = np.append(step_200.mean(), overall_avgs)
    overall_avgs = np.append(step_175.mean(), overall_avgs)
    overall_avgs = np.append(step_150.mean(), overall_avgs)
    overall_avgs = np.append(step_125.mean(), overall_avgs)
    overall_avgs = np.append(step_100.mean(), overall_avgs)
    overall_avgs = np.append(step_75.mean(), overall_avgs)
    overall_avgs = np.append(step_50.mean(), overall_avgs)
    overall_avgs = np.append(step_25.mean(), overall_avgs)
    overall_avgs = overall_avgs.flatten()

    overall_stds = np.array([])
    overall_stds = np.append(stds_300.mean(), overall_stds)
    overall_stds = np.append(stds_275.mean(), overall_stds)
    overall_stds = np.append(stds_250.mean(), overall_stds)
    overall_stds = np.append(stds_225.mean(), overall_stds)
    overall_stds = np.append(stds_200.mean(), overall_stds)
    overall_stds = np.append(stds_175.mean(), overall_stds)
    overall_stds = np.append(stds_150.mean(), overall_stds)
    overall_stds = np.append(stds_125.mean(), overall_stds)
    overall_stds = np.append(stds_100.mean(), overall_stds)
    overall_stds = np.append(stds_75.mean(), overall_stds)
    overall_stds = np.append(stds_50.mean(), overall_stds)
    overall_stds = np.append(stds_25.mean(), overall_stds)
    overall_stds = overall_stds.flatten()

    df = pd.DataFrame({'x': range(0, 30), 'y1': step_25, 'y2': step_50, 'y3': step_75, 'y4': step_100,
                       'y5': step_125, 'y6': step_150, 'y7': step_175, 'y8': step_200, 'y9': step_225,
                       'y10': step_250, 'y11': step_275, 'y12': step_300, 'y13': chance})

    x = range(25, 325, 25)

    if which_plot == 0:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot('x', 'y13', data=df, marker='', color='black', linewidth=3, label='baseline')
        ax.plot('x', 'y1', data=df, marker='', color='olive', linewidth=1.5, label='25_frames')
        ax.plot('x', 'y2', data=df, marker='', color='blue', linewidth=1.5, label='50_frames')
        ax.plot('x', 'y3', data=df, marker='', color='orange', linewidth=1.5, label='75_frames')
        ax.plot('x', 'y4', data=df, marker='', color='purple', linewidth=1.5, label='100_frames')
        ax.plot('x', 'y5', data=df, marker='', color='red', linewidth=1.5, label='125_frames')
        ax.plot('x', 'y6', data=df, marker='', color='limegreen', linewidth=1.5, label='150_frames')
        ax.plot('x', 'y7', data=df, marker='', color='yellow', linewidth=1.5, label='175_frames')
        ax.plot('x', 'y8', data=df, marker='', color='pink', linewidth=1.5, label='200_frames')
        ax.plot('x', 'y9', data=df, marker='', color='skyblue', linewidth=1.5, label='225_frames')
        ax.plot('x', 'y10', data=df, marker='', color='darkorange', linewidth=1.5, label='250_frames')
        ax.plot('x', 'y11', data=df, marker='', color='cyan', linewidth=1.5, label='275_frames')
        ax.plot('x', 'y12', data=df, marker='', color='magenta', linewidth=1.5, label='300_frames')
        plt.xticks(np.arange(0, 30, 2))
        plt.yticks(np.arange(0.00, 1.01, 0.05))
        ax.legend(loc=3, ncol=2)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Test Number')
        ax.set_ylabel('Average Accuracy (%)')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: testing different time step values, reduced set: rw_lsi")
        plt.ylim(0.00, 1.00)
    elif which_plot == 1:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_avgs, marker='', color='red', linewidth=1.5, label='average accuracy')
        plt.xticks(np.arange(25, 300.1, 25))
        plt.yticks(np.arange(0.00, 1.01, 0.05))
        ax.legend(loc=3)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Number of frames per step')
        ax.set_ylabel('Average Accuracy (%)')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: testing different time step values, reduced set: rw_lsi")
        plt.ylim(0.00, 1.00)
    elif which_plot == 2:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_stds, marker='', color='blue', linewidth=1.5, label='average stdev')
        plt.xticks(np.arange(25, 300.1, 25))
        plt.yticks(np.arange(0.00, 0.10, 0.01))
        ax.legend(loc=3)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Number of frames per step')
        ax.set_ylabel('Average stdev')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: testing different time step values, reduced set: rw_lsi")
        plt.ylim(0.01, 0.09)

    plt.show()

    print("The overall average accuracy for 25 frames was: ", step_25.mean())
    print("for 50 frames: ", step_50.mean())
    print("for 75 frames: ", step_50.mean())
    print("for 100 frames: ", step_100.mean())
    print("for 125 frames: ", step_125.mean())
    print("for 150 frames: ", step_150.mean())
    print("for 175 frames: ", step_175.mean())
    print("for 200 frames: ", step_200.mean())
    print("for 225 frames: ", step_225.mean())
    print("for 250 frames: ", step_250.mean())
    print("for 275 frames: ", step_275.mean())
    print("for 300 frames: ", step_300.mean())
    print("\n\n")
    print("The average stderr for 25 frames was: ", stds_25.mean())
    print("for 50 frames: ", stds_50.mean())
    print("for 75 frames: ", stds_50.mean())
    print("for 100 frames: ", stds_100.mean())
    print("for 125 frames: ", stds_125.mean())
    print("for 150 frames: ", stds_150.mean())
    print("for 175 frames: ", stds_175.mean())
    print("for 200 frames: ", stds_200.mean())
    print("for 225 frames: ", stds_225.mean())
    print("for 250 frames: ", stds_250.mean())
    print("for 275 frames: ", stds_275.mean())
    print("for 300 frames: ", stds_300.mean())


# This method was used to visualise the tests for the different sizes of hidden layers. Its function
# basically stores the variables for the single-layer line, two-layer line, and three-layer line separately.
# Then, depending upon the which_plot value passed it will plot either the accuracy progression with increasing
# hidden layer sizes (0), or the standard deviation progresssion with increasing hidden layer sizes (1).
def visualise_hidden_layer_comparison(which_plot):
    # Create arrays for looping through file names
    num_nodes = ["50", "100", "150", "200", "250", "300", "350", "400", "450", "500", "550", "600"]
    f_names = ["single_layer_tests/", "two_layer_tests/", "three_layer_tests/"]
    num_nodes_avgs = []
    num_nodes_stds = []
    num_nodes_avgs_2 = []
    num_nodes_stds_2 = []
    num_nodes_avgs_3 = []
    num_nodes_stds_3 = []

    # Load variables from relevant pickle files.
    for x in range(3):
        for i in range(len(num_nodes)):
            f_name = "pickle_files/" + f_names[x] + "cross_val_avgs_layer_size_" + num_nodes[i]
            var = []
            if x == 0:
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_avgs.append(var)
                    f.close()

                f_name = "pickle_files/" + f_names[x] + "cross_val_stdev_layer_size_" + num_nodes[i]
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_stds.append(var)
                    f.close()
            elif x == 1:
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_avgs_2.append(var)
                    f.close()

                f_name = "pickle_files/" + f_names[x] + "cross_val_stdev_layer_size_" + num_nodes[i]
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_stds_2.append(var)
                    f.close()
            elif x == 2:
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_avgs_3.append(var)
                    f.close()

                f_name = "pickle_files/" + f_names[x] + "cross_val_stdev_layer_size_" + num_nodes[i]
                with open(f_name, 'rb') as f:
                    var = pickle.load(f)
                    num_nodes_stds_3.append(var)
                    f.close()

    # For each line type, store the values in a NumPy array and flatten the data to 1 dimension
    overall_avgs = np.array([])
    overall_stds = np.array([])
    num_nodes_avgs = np.asarray(num_nodes_avgs)
    num_nodes_stds = np.asarray(num_nodes_stds)

    for i in range(11, -1, -1):
        overall_avgs = np.append(num_nodes_avgs[i].mean(), overall_avgs)

    overall_avgs = overall_avgs.flatten()

    for i in range(11, -1, -1):
        overall_stds = np.append(num_nodes_stds[i].mean(), overall_stds)

    overall_stds = overall_stds.flatten()

    overall_avgs_2 = np.array([])
    overall_stds_2 = np.array([])
    num_nodes_avgs_2 = np.asarray(num_nodes_avgs_2)
    num_nodes_stds_2 = np.asarray(num_nodes_stds_2)

    for i in range(11, -1, -1):
        overall_avgs_2 = np.append(num_nodes_avgs_2[i].mean(), overall_avgs_2)

    overall_avgs_2 = overall_avgs_2.flatten()

    for i in range(11, -1, -1):
        overall_stds_2 = np.append(num_nodes_stds_2[i].mean(), overall_stds_2)

    overall_stds_2 = overall_stds_2.flatten()

    overall_avgs_3 = np.array([])
    overall_stds_3 = np.array([])
    num_nodes_avgs_3 = np.asarray(num_nodes_avgs_3)
    num_nodes_stds_3 = np.asarray(num_nodes_stds_3)

    for i in range(11, -1, -1):
        overall_avgs_3 = np.append(num_nodes_avgs_3[i].mean(), overall_avgs_3)

    overall_avgs_3 = overall_avgs_3.flatten()

    for i in range(11, -1, -1):
        overall_stds_3 = np.append(num_nodes_stds_3[i].mean(), overall_stds_3)

    overall_stds_3 = overall_stds_3.flatten()

    x = range(50, 650, 50)

    if which_plot == 0:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_avgs, marker='', color='blue', linewidth=1.5, label='single layer')
        ax.plot(x, overall_avgs_2, marker='', color='black', linewidth=1.5, label='two layers')
        ax.plot(x, overall_avgs_3, marker='', color='skyblue', linewidth=1.5, label='three layers')
        plt.xticks(np.arange(50, 600.1, 50))
        plt.yticks(np.arange(0.00, 1.01, 0.05))
        ax.legend(loc=3)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Number of nodes per layer')
        ax.set_ylabel('Average Accuracy (%)')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: 200, reduced set: rw_lsi\n"
                  "Testing different numbers of layers")
        plt.ylim(0.00, 1.01)
    elif which_plot == 1:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_stds, marker='', color='blue', linewidth=1.5, label='single layer')
        ax.plot(x, overall_stds_2, marker='', color='black', linewidth=1.5, label='two layers')
        ax.plot(x, overall_stds_3, marker='', color='skyblue', linewidth=1.5, label='three layers')
        plt.xticks(np.arange(50, 600.1, 50))
        plt.yticks(np.arange(0.00, 0.31, 0.05))
        ax.legend(loc=7)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Number of nodes per layer')
        ax.set_ylabel('Average stdev')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: 200, reduced set: rw_lsi\n"
                  "Testing different numbers of layers")
        plt.ylim(0.00, 0.31)

    plt.show()

# Works essentially the same as visualise_hidden_layer_comparison() except with only one line to deal with.
def visualise_max_iter_comparison(which_plot):
    max_iter = ["500", "1000", "1500", "2000", "2500", "3000", "3500", "4000",
                "4500", "5000", "5500", "6000", "6500", "7000", "7500", "8000",
                "8500", "9000", "9500", "10000"]
    max_iter_avgs = np.array([])
    max_iter_stds = np.array([])

    for i in range(len(max_iter)):
        f_name = "pickle_files/max_iterations_tests/cross_val_avgs_its_" + max_iter[i]
        with open(f_name, 'rb') as f:
            var = pickle.load(f)
            max_iter_avgs = np.append(var, max_iter_avgs)
            f.close()

        f_name = "pickle_files/max_iterations_tests/cross_val_stdev_its_" + max_iter[i]
        with open(f_name, 'rb') as f:
            var = pickle.load(f)
            max_iter_stds = np.append(var, max_iter_stds)
            f.close()

    overall_avgs = np.array([])
    overall_stds = np.array([])

    for i in range(19, -1, -1):
        overall_avgs = np.append(max_iter_avgs[i].mean(), overall_avgs)

    overall_avgs = overall_avgs.flatten()

    for i in range(19, -1, -1):
        overall_stds = np.append(max_iter_stds[i].mean(), overall_stds)

    overall_stds = overall_stds.flatten()

    x = range(500, 10500, 500)

    if which_plot == 0:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_avgs, marker='', color='blue', linewidth=1.5, label='Mean Accuracies')
        plt.xticks(np.arange(500, 10500.1, 1000))
        plt.yticks(np.arange(0.00, 1.01, 0.05))
        ax.legend(loc=3)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Max Iterations')
        ax.set_ylabel('Average Accuracy (%)')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: 200, reduced set: rw_lsi\n"
                  "Testing different max iteration settings")
        plt.ylim(0.00, 1.01)
    elif which_plot == 1:
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x, overall_stds, marker='', color='blue', linewidth=1.5, label='single layer')
        plt.xticks(np.arange(500, 10500.1, 1000))
        plt.yticks(np.arange(0.00, 0.31, 0.05))
        ax.legend(loc=7)
        ax.grid(color='black', linestyle='-', linewidth=0.2)
        ax.set_xlabel('Max Iterations')
        ax.set_ylabel('Average stdev')
        plt.title("K-fold Cross Validation Scores\n "
                  "Time step: 200, reduced set: rw_lsi\n"
                  "Testing different max iteration settings")
        plt.ylim(0.00, 0.31)

    plt.show()


# For the purposes of demonstration, only this method needs to be run, providing an example of
# the type of plots which were created during the experiments stage.
visualise_full_vs_red_vs_raw()

print("Visualise.py is complete, test successful!")
