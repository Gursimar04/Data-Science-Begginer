import pandas as pd


def excel_to_csv(source_path, destination_path):
    # Read all sheets of an Excel file and return each sheet as an individual csv file
    print(f'Loading {source_path} into pandas')
    file = pd.ExcelFile("C:/Users/Predator/Desktop/input_file.xls")
    df = pd.DataFrame()
    for idx, name in enumerate(file.sheet_names):
        print(f'Exporting sheet number {idx + 1}: {name}')
        sheet = file.parse(name)
        sheet.to_csv(destination_path + "/Sheet" + str(idx + 1) + ".csv", index=None, header=True)


excel_to_csv("C:/Users/Predator/Desktop/input_file.xls", "C:/Users/Predator/Desktop/OutputFolder")

# Function which takes in a path of source file and path for destination a
# It coverts all sheets of an excel file to a csv
