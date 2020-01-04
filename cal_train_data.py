def Cal_Train_data(num_per_azi,src_start,room_list,inter_dir):
    
    src_path_file_path='TIMIT_wav_path_train_shuffled.txt'
    with open(src_path_file_path,'r') as src_path_file:
        src_paths = src_path_file.readlines()
    
    azi_range = np.asarray([-8,-6,-4,-2,-1,1,2,4,6,8],dtype=np.int16)
    SNRs = [0,10,20]

    log_file = open('inter_record.txt','w')

    for room_name in room_list:
        brirs_room = Get_BRIRs(room_name)
        src_wav_count = src_start #
        for azi_tar in range(8,29):
            azi_tar_dir = os.path.join(inter_dir,room_name,str(azi))
            if not os.path.exists(azi_tar_dir):
                os.makedirs(dst_record_dir)

            for src_i_tar in range(src_wav_count,src_wav_count+num_per_azi):
                for azi_inter in (azi_range+azi_tar):
                    for SNR in SNRs:
                    # randome select interferent sentence
                        src_i_inter = src_i_tar
                        while src_i_inter != src_i_tar:
                            src_i_inter = np.random.randint(0,len(src_paths),1)

                        src_path_inter = src_paths[src_i_inter].strip()
                        log_file.write('room:{}  azi_tar:{}  src_i_tar:{}  \\\
                                        azi_inter:{} SNR:{}  \\\
                                        inter_src_path:{}\n'.format(room_name,azi_tar,src_i_tar,
                                                                   azi_inter,SNR,
                                                                   src_path_inter))
                        record_path_inter = os.path.join(azi_tar_dir,
                                                         '{}_{}_{}.wav'.format(src_i_tar,
                                                                                azi_inter,
                                                                                SNR))
                        Syn_record(src_path_inter,
                                   record_path_inter,
                                   brirs_room[azi_inter])

            src_wav_count=src_wav_count+num_per_azi
