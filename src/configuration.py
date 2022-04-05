import configparser

CONFIG_PATH = "../config/"
FILE_NAME = "config.ini"


def config_deeplearing():
    """[To use config/config.ini properly]

    Returns:
        [dict]: [config dict according to config/config.ini file]
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH + FILE_NAME)
    return config


if __name__ == "__main__":
    config = config_deeplearing()
    print("hello world")
    print("Path: ", config["PATH"]["PATH"])
    print("Learing Rate: ", config["PARAMETERS"]["LR"])
