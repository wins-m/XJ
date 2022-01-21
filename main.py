"""
 (created by swmao on Jan. 10th)

"""
import time
import yaml

# %% Main
if __name__ == '__main__':
    time_start = time.time()

    conf = yaml.safe_load(open('config.yaml', encoding='utf-8'))

    # data to local
    # from data.get_data import get_data
    # get_data(conf)

    # calculate pe residual
    # from data.pe_surprise import cal_pe_surprise
    # cal_pe_surprise(conf)

    # upload local tables
    # from data.save_remote import save_remote
    # save_remote(conf)

    print(f'total time cost {time.time() - time_start : .3f} s\n')