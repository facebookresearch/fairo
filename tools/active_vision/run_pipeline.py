from find_respawn_loc import find_spawn_loc
import submitit
import argparse
import os


find_spawn_loc(baseline_root, outdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for running active vision pipeline")
    parser.add_argument(
        "--data_path",
        help="path where scene data is being stored",
        type=str,
    )
    parser.add_argument("--job_folder", type=str, default="", help="")
    parser.add_argument("--comment", type=str)
    parser.add_argument("--slurm", action="store_true", default=False, help="Run the pipeline on slurm, else locally")

    args = parser.parse_args()

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=os.path.join(args.job_folder, 'slurm_logs/%j'))
    # set timeout in min, and partition for running the job
    executor.update_parameters(
        slurm_partition="learnfair", #"learnfair", #scavenge
        timeout_min=2000,
        mem_gb=256,
        gpus_per_node=4,
        tasks_per_node=1, 
        cpus_per_task=8,
        additional_parameters={
            "mail-user": f"{os.environ['USER']}@fb.com",
            "mail-type": "all",
        },
        slurm_comment="Droidlet Active Vision Pipeline"
    )

    job = executor.submit(find_spawn_loc, args.data_path, args.job_dir)


    # gtps = set()
    # for gt in range(5, 15, 5):
    #     for p in range(0, 30, 5):
    #         gtps.add((gt,p))

    # for gt in range(5, 30, 5):
    #     for p in range(0,15,5):
    #         gtps.add((gt,p))

    # gtps = sorted(list(gtps))
    # print(len(gtps), gtps)

    # # Ten settings for quick turnaround
    # gtps = set()
    # for gt in range(5, 10, 5):
    #     for p in range(0, 30, 5):
    #         gtps.add((gt,p))

    # for gt in range(5, 30, 5):
    #     for p in range(5,10,5):
    #         gtps.add((gt,p))

    # gtps = sorted(list(gtps))
    # print(len(gtps), gtps)

    # jobs = []
    # if args.slurm:
    #     with executor.batch():
    #         for traj in range(args.num_traj+1):
    #             traj_path = os.path.join(args.data_path, str(traj))
    #             if os.path.isdir(traj_path):
    #                 s = SampleGoodCandidates(traj_path, is_annot_validfn)
    #                 for gt, p in gtps: 
    #                     src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
    #                     if src_img_ids is not None and len(src_img_ids) > 0:
    #                         job = executor.submit(
    #                             _runner, traj, gt, p, args.active, args.data_path, args.job_folder, args.num_train_samples, src_img_ids
    #                         )
    #                         jobs.append(job)
    #     log_job_start(args, jobs)
    #     print(f'{len(jobs)} jobs submitted')
    
    # else:
    #     print('running locally ...')
    #     for traj in range(args.num_traj+1):
    #         traj_path = os.path.join(args.data_path, str(traj))
    #         s = SampleGoodCandidates(traj_path, is_annot_validfn)
    #         for gt in range(5, 10, 5):
    #             for p in range(5, 10, 5): # only run for fixed gt locally to test
    #                 src_img_ids = s.get_n_candidates(gt, good=True, evenly_spaced=True)
    #                 _runner(traj, gt, p, args.active, args.data_path, args.job_folder, args.num_train_samples, src_img_ids)