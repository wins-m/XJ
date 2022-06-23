"""
 (created by swmao on Jan. 10th)

"""
import time

# %% Main
if __name__ == '__main__':
    import yaml
    conf = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    time_start = time.time()

    # * data to local *
    # from data.get_data import get_data
    # get_data(conf)

    # * calculate pe residual *
    # from data.pe_surprise import cal_pe_surprise
    # cal_pe_surprise(conf)

    # * upload local tables *
    # from data.save_remote import save_remote
    # save_remote(conf)

    from BarraPCA.optimize import optimize
    optimize(conf, mkdir_force=True, process_num=4)
    # from BarraPCA.opt_res_ana import opt_res_ana
    # opt_res_ana(conf, test=False)

    print(f'total time cost {time.time() - time_start : .3f} s\n')
