from configuration import SysConfAdvSW
from itertools import chain
from utility import EstimateHyperparameters
"""
    Print in file parameters.txt the estimate of the kernel parameters for each function of the advertising experiment
"""


def write_param(string_param):
    file1 = open("parameters.txt", "w")
    file1.write(string_param)
    file1.close()


if __name__ == '__main__':

    config = SysConfAdvSW.SysConfAdvSW("/home/mattia/PyProjects/DIA-Project/configuration/")
    func_list = list(chain(*config.function()))
    sigma = config.init_noise()
    name = config.function_name()
    string = ''

    for i in range(0, len(func_list)):
        estimate = EstimateHyperparameters.EstimateHyperparameters(func_list[i], 10, 200, sigma)
        estimate.fit(*(estimate.sample()))
        if i % 3 == 0:
            string = string + str(name[i:3]) + "\n"
        string = string + str(estimate.get_parameters()) + "\n"

    write_param(string)