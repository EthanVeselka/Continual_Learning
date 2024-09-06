import re
import numpy as np


def parse_avg_sensitivity_specificity(task, file_path):
    with open(file_path, "r") as file:
        data = file.read()

    # Regular expressions to capture sensitivity and specificity arrays
    if task == "los" or task == "phen":
        sensitivity_pattern = r"Avg Sensitivity:\s*(\[\[[\s\S]+?\]\])"
        specificity_pattern = r"Avg Specificity:\s*(\[\[[\s\S]+?\]\])"
    else:
        sensitivity_pattern = r"Avg Sensitivity:\s*(\[\[?[\s\S]+?\]?\])"
        specificity_pattern = r"Avg Specificity:\s*(\[\[?[\s\S]+?\]?\])"

    # Extract sensitivity and specificity arrays
    sensitivity_match = re.search(sensitivity_pattern, data)
    specificity_match = re.search(specificity_pattern, data)

    if sensitivity_match and specificity_match:
        # Clean and convert the extracted text into lists of numbers
        sensitivity_str = sensitivity_match.group(1).replace("\n", " ").strip()
        specificity_str = specificity_match.group(1).replace("\n", " ").strip()
        # print("str: ", sensitivity_str)
        # print("str: ", specificity_str)

        # Split by spaces and convert to NumPy array (2D)
        # Manually convert space-separated values to 2D list (row by row)
        def convert_to_array(array_str):
            rows = array_str.replace("]  [", "], [").split(
                ", "
            )  # Split into rows by double space ]  [
            array = []
            for row in rows:
                # Remove any extra brackets and split by spaces
                clean_row = row.replace("[", "").replace("]", "").strip()
                array.append([float(val) for val in clean_row.split()])
            return array

        sensitivity_array = convert_to_array(sensitivity_str)
        specificity_array = convert_to_array(specificity_str)

        # print("\n")
        # print("Avg Sensitivity (as array):")
        # print(sensitivity_array)

        # print("\nAvg Specificity (as array):")
        # print(specificity_array)

        return (sensitivity_array, specificity_array)


# Call the function with the path to your text file
# file_path = "./los/south/sens_test.txt"
# parse_avg_sensitivity_specificity(file_path)


def generate_latex_table(task, results):
    if task == "dec":
        string = "Decompensation"
        class_names = ["MIMIC-III", "eICU Region"]
    elif task == "ihm":
        string = "In-hospital Mortality"
        class_names = ["MIMIC-III", "eICU Region"]
    elif task == "phen":
        string = "Phenotyping"
        class_names = [
            "Acute and unspecified renal failure",
            "Acute cerebrovascular disease",
            "Acute myocardial infarction",
            "Cardiac dysrhythmias",
            "Chronic kidney disease",
            "Chronic obstructive pulmonary disease and bronchiectasis",
            "Complications of surgical procedures or medical care",
            "Conduction disorders",
            "Congestive heart failure; nonhypertensive",
            "Coronary atherosclerosis and other heart disease",
            "Diabetes mellitus with complications",
            "Diabetes mellitus without complication",
            "Disorders of lipid metabolism",
            "Essential hypertension",
            "Fluid and electrolyte disorders",
            "Gastrointestinal hemorrhage",
            "Hypertension with complications and secondary hypertension",
            "Other liver diseases",
            "Other lower respiratory disease",
            "Other upper respiratory disease",
            "Pleurisy; pneumothorax; pulmonary collapse",
            "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
            "Respiratory failure; insufficiency; arrest (adult)",
            "Septicemia (except in labor)",
            "Shock",
        ]
    elif task == "los":
        string = "Length of Stay"
        class_names = [
            "Less than 24 hours",
            "Over 24 hours",
            "48 hours",
            "72 hours",
            "96 hours",
            "120 hours",
            "144 hours",
            "168 hours",
            "Over 1 week",
            "Over 2 weeks",
        ]
    num_classes = len(
        class_names
    )  # Number of classes will define the number of columns for Sensitivity/Specificity

    # Create the header for class columns for sensitivity and specificity
    class_columns = (
        " & ".join([f"Sensitivity {cls}" for cls in class_names])
        + " & "
        + " & ".join([f"Specificity {cls}" for cls in class_names])
    )

    if task == "dec" or task == "ihm":
        latex_code = (
            "\\begin{table}[h!]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{cc" + "".join(["c"] * num_classes * 2) + "}\n"
            "\\toprule\n"
            f"\\multicolumn{{6}}{{c}}{{\\textbf{{{string}}}}} \\\\\n"
            "\\midrule\n"
        )
        latex_code += f"Method & Region & {class_columns} \\\\\n\\midrule\n"

        # Iterate over regions and methods
        for region, methods in results.items():
            j = 0
            for method, arrays in methods.items():
                if method == "base":
                    method = "Baseline"
                elif method == "ewc":
                    method = "EWC"
                elif method == "trrep":
                    method = "Replay"
                elif method == "adjrep":
                    method = "Adj Replay"
                elif method == "comb":
                    method = "Combined"

                sensitivity, specificity = arrays

                # Combine the sensitivity and specificity arrays for each class
                sensitivity = np.array(sensitivity).flatten()
                specificity = np.array(specificity).flatten()
                sens_values = " & ".join([f"{round(val, 3)}" for val in sensitivity])
                spec_values = " & ".join([f"{round(val, 3)}" for val in specificity])

                # Add the data row for each region and method
                if j == 0:
                    latex_code += f"{method} & \\multirow{{5}}{{*}}{{{region.capitalize()}}} &  {sens_values} & {spec_values} \\\\\n"
                else:
                    latex_code += f"{method} & & {sens_values} & {spec_values} \\\\\n"
                j += 1
            latex_code += "\\midrule\n"
        latex_code += "\\end{tabular}\n" "}\n" "\\end{table}\n"

    else:
        latex_code = (
            "\\begin{longtable}{lcc}\n"
            "\\toprule\n"
            f"\\multicolumn{{3}}{{c}}{{\\textbf{{{string}}}}} \\\\\n"
            "\\toprule\n"
            "\\endfirsthead\n\n"
            "\\midrule\n"
            "\endhead\n\n"
        )
        # Iterate over regions and methods
        for region, methods in results.items():
            r = 0
            latex_code += f"\\multicolumn{{3}}{{c}}{{\\textbf{{Region: {region.capitalize()}}}}} \\\\\n"
            for method, arrays in methods.items():
                sensitivity, specificity = arrays
                if method == "base":
                    method = "Baseline"
                elif method == "ewc":
                    method = "EWC"
                elif method == "trrep":
                    method = "Replay"
                elif method == "adjrep":
                    method = "Adj Replay"
                elif method == "comb":
                    method = "Combined"

                latex_code += f"\\multicolumn{{3}}{{c}}{{\\textbf{{Method: {method}}}}} \\\\\n\\midrule\n"

                for i in range(num_classes):
                    class_name = class_names[i]
                    sens_value = f"{round(sensitivity[r][i], 3)}"
                    spec_value = f"{round(specificity[r][i], 3)}"

                    # Add row for each class
                    latex_code += f"{class_name} & {sens_value} & {spec_value} \\\\\n"
                latex_code += "\\midrule\n"
            latex_code += "\\midrule\n"
            r += 1
        latex_code += "\\end{longtable}\n"

    return latex_code


# Data structure to hold the results for each task
results = {"ihm": {}, "phen": {}, "dec": {}, "los": {}}

for task in ["ihm", "phen", "dec", "los"]:  # ["ihm", "phen", "dec", "los"]:
    for region in ["south", "midwest", "west", "northeast"]:
        results[task][region] = {}
        for method in ["base", "ewc", "trrep", "adjrep", "comb"]:
            bs = 3500 if (task == "dec" or task == "los") else 500
            bs = 0 if method == "base" else bs
            file = f"./{task}/{region}/{task}_{bs}_{method}_CE_BiLSTM.txt"
            sensitivity, specificity = parse_avg_sensitivity_specificity(task, file)

            results[task][region][method] = (sensitivity, specificity)


# Generate LaTeX tables for each task
for task in ["ihm", "phen", "dec", "los"]:
    latex_table = generate_latex_table(task, results[task])
    file_name = f"./{task}/metrics.tex"
    with open(file_name, "a") as file:
        file.write(latex_table)
        file.write("\n\n")
