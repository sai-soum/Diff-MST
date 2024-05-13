import csv
import sys
import pandas as pd
import os

if __name__ == "__main__":
    csv_paths = ["/Users/svanka/Codes/Diff-MST/outputs/listen/AF/by-my-side.csv",
                 "/Users/svanka/Codes/Diff-MST/outputs/listen/AF/ecstasy.csv"
    ]

    for csv_path in csv_paths:
        print(os.path.basename(csv_path).replace('.csv', ''))
        #read csv using pandas
        df = pd.read_csv(csv_path)
        #find the row with the lowest loss
        lowest_loss = df['net_AF_loss'].min()
        #find the method,audio_section  with the lowest loss
        lowest_loss_row = df.loc[df['net_AF_loss'] == lowest_loss]
        print(lowest_loss_row)
