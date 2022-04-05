import os


def csv_generator(dic_predictions: dict, name: str, path: str = "../outputs/"):
    """[Generate a CSV file from a dict which contains predictions from an input image]

    Args:
        dic_predictions (dict): [image name, class prediction]
        name (str): [Name of the CSV file]
        path (str, optional): [Path to store the CSV file]. Defaults to "../outputs/".
    """
    # Check if output path exits
    if not os.path.exists(path):
        print("Creating output path : ", path)
        os.makedirs(path)
    # Compute the number of the CSV prediction from the kind of the model
    lcsv = os.listdir(path)
    n = 0
    for csv in lcsv:
        if csv.startswith(name):
            n += 1
    # Open file to write lines
    with open(path + name + "_" + str(n) + ".csv", "w") as csvfile:
        csvfile.write(
            "imgname,label"
        )  # First line of the csv file -> imposed by challenge
        for i in dic_predictions.keys():
            csvfile.write("\n")
            csvfile.write(i + "," + dic_predictions[i])


def csv_to_dic(name: str):
    """[create a dictionnary from a csv file]

    Args:
        name (str): [name of the csv file]

    Returns:
        [dict]: [dictionnary creating from csv file]
    """
    dic = {}
    with open(name, "r") as csvfile:
        line = csvfile.readline()
        # don't take the first line because it's 'imgname,label'
        conti = True
        while conti:
            line = csvfile.readline()
            if line != "":
                image_num = line.split(",")[0].split(".")[0]
                plankton_class = line.split(",")[1].strip()
                dic[image_num] = plankton_class
            else:
                conti = False
    return dic


def correct_dic(wrong_dic: dict, ref_tesloader, testset, online=True):
    """[Apply corrections to a dictionnary where classes are not well named]

    Args:
        wrong_dic (dict): [dictionnary to apply corrections]
        tesloader ([type]): [reference testloader]

    Returns:
        [dict]: [dictionnary with corrections]
    """
    dic_true = {}
    for i in range(len(ref_tesloader)):
        sample = testset[i]
        path_to_sample = testset.samples[i][0]
        name = path_to_sample.split("/")[-1]
        if online:
            dic_true[name] = wrong_dic[str(i) + ".jpg"]
        else:
            dic_true[name] = wrong_dic[str(i)]
    return dic_true
