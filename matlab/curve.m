function [] = curve(nbit, db_name)
    %% init and load
    addpath('utils/');
    fprintf('Load hashcode from Our-model.\n');
    save_path = sprintf('../Hashcode/BSTH_%d_%s_bits.mat', nbit, db_name);
    load(save_path);

    %% compact hashcode
    B_trn = compactbit(retrieval_B > 0);
    B_tst = compactbit(val_B > 0);

    Dhamm = hammingDist(B_tst, B_trn)';

    [~, Dhamm_index] = sort(Dhamm, 1);
    [Pre, Rec, MAP] = fast_PR_MAP(int32(cateTrainTest), int32(Dhamm_index));
    [Pre_top, Rec_top, MAP_top] = get_PR_top(Pre, Rec, MAP);

    result_name = ['../Result/' db_name '_' num2str(nbit) '_result' '.mat'];
    save(result_name, 'MAP_top', 'Pre_top', 'Rec_top', 'MAP', 'Pre', 'Rec');

    fprintf('[%s-%s] MAP@N = %.4f, MAP@50 = %.4f\n', db_name, num2str(nbit), MAP(end), MAP(50));

    %% save to file
    name = ['../Result/' db_name '.txt'];
    fid = fopen(name, 'a+');
    fprintf(fid, '[%s-%s] MAP@N = %.4f, MAP@50 = %.4f\n', db_name, num2str(nbit), MAP(end), MAP(50));
end

