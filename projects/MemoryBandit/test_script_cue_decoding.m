%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_cue/');

ROIs = {'HPC', 'OFC', 'dmPFC', 'dlPFC', 'ACC', 'AMY'};  % list of ROIs
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

dependent_var = 'memory_response';  % hits, misses, false_alarms, correct_rejections
downsample_factor = 5;            % match your Python version
win_ms = 200;                     % smoothing window in ms
permutation = 0;
nPerm = 1000;                      % number of permutations for significance
alpha = 0.05;                     % cluster-level significance
thresh = 0.75;
min_cluster_pts = round(0.100 * 100);  % e.g., fs=250 => 38 points

%% --- LOAD TABLES ONCE ---
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- LOOP OVER ROIs ---
for r = 1:numel(ROIs)
    roi = ROIs{r};
    fprintf('Processing ROI: %s\n', roi);
    
    % Preallocate arrays for all subjects
    all_acc_subjects = [];
    all_perm_subjects = [];
    
    %% --- LOOP OVER SUBJECTS ---
    for s = 1:numel(subjects)
        subj = subjects{s};
        fprintf('  Subject: %s\n', subj);
        
        %% --- SELECT SUBJECT DATA ---
        subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
%         subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_mem), :);

        %% Create memory response labels from boolean columns
        % Convert cell arrays of 'True'/'False' strings to logical
        hits = strcmp(subj_combined_df.hits, 'True');
        misses = strcmp(subj_combined_df.misses, 'True');
        false_alarms = strcmp(subj_combined_df.false_alarms, 'True');
        correct_rejections = strcmp(subj_combined_df.correct_rejections, 'True');
        
        % Initialize with NaN
        memory_response = nan(height(subj_combined_df), 1);
        
        % Assign labels: 1=hit, 2=miss, 3=false_alarm, 4=correct_rejection
        memory_response(hits) = 1;
        memory_response(misses) = 2;
        memory_response(false_alarms) = 3;
        memory_response(correct_rejections) = 4;
        
        % Add to dataframe
        subj_combined_df.memory_response = memory_response;
        
        % Remove any trials without a valid memory response
        subj_combined_df = subj_combined_df(~isnan(subj_combined_df.memory_response), :);

        %% Trial mapping and remove fast RTs
        retrieval_trials = subj_combined_df.trials_mem;
        mask_rt = subj_combined_df.recog_rt >= 0.3;
        retrieval_trials = retrieval_trials(mask_rt);

        if isempty(retrieval_trials)
            warning('No trials for subject %s', subj);
            continue;
        end

        %% Load electrode info
        elec_df = readtable(fullfile(base_dir, 'projects','guLab','Salman','EphysAnalyses', subj, 'Day1_reref_elec_df'));
        roi_electrodes = elec_df.label(strcmp(elec_df.salman_region, roi));
        if isempty(roi_electrodes)
            warning('No electrodes found for %s in %s', subj, roi);
            continue;
        end

        %% Load cue TFR data
        filepath = fullfile(base_dir,'projects','guLab','Salman','EphysAnalyses',subj,'scratch','TFR');
        cue = load(fullfile(filepath, 'cue_on-tfr.mat'));
        cue_tfr.powspctrm = cue.powspctrm;
        cue_tfr.freq = cue.freqs;
        cue_tfr.time = cue.times;
        cue_tfr.label = cue.ch_names;
        cue_tfr.fsample = cue.sfreq;

        %% Select ROI electrodes
        [~, idx_cue] = ismember(roi_electrodes, cue_tfr.label);
        cue_tfr.powspctrm = cue_tfr.powspctrm(:, idx_cue, :, :);

        %% Extract HFA 70–200 Hz
        hfa_mask = cue_tfr.freq >= 70 & cue_tfr.freq <= 201;
        cue_hfa = squeeze(mean(cue_tfr.powspctrm(:, :, hfa_mask, :), 3));  % trials x electrodes x time
        Ndims = ndims(cue_hfa); 
        if Ndims < 3 
            cue_hfa = reshape(cue_hfa, [size(cue_hfa, 1), 1, size(cue_hfa, 2)]); 
        end
        
        %% --- SMOOTH (VECTORIZE) ---
        fs = cue_tfr.fsample;
        win_size = round(fs * (win_ms/1000));
        cue_hfa_smooth = movmean(cue_hfa, win_size, 3);  % vectorized smoothing

        %% Select retrieval trials
        cue_hfa_smooth = cue_hfa_smooth(retrieval_trials,:,:);

        %% Downsample
        cue_hfa_ds = cue_hfa_smooth(:,:,1:downsample_factor:end);

        %% Define labels
        y = subj_combined_df.(dependent_var);
        y = y(mask_rt);
        chancelevel = 0.25;  % 4 classes
        
        % Check class balance
        fprintf('    Memory response distribution: Hit=%d, Miss=%d, FA=%d, CR=%d\n', ...
            sum(y==1), sum(y==2), sum(y==3), sum(y==4));
        
        y = double(y);

        %% --- MVPA-LIGHT CONFIGURATION ---
        cfg = [];
        cfg.classifier = 'multiclass_lda'; 
        cfg.metric = 'acc'; 
        cfg.prior = 'uniform';  % equal prior for all 4 classes
        cfg.cv = 'kfold'; 
        cfg.k = 5; 
        cfg.repeat = 10; 
        cfg.feedback = 0;

        %% --- TIME-RESOLVED DECODING ---
        [perf, results] = mv_classify_across_time(cfg, cue_hfa_ds, y);

        % Preallocate arrays if first subject
        if isempty(all_acc_subjects)
            num_timepoints = size(perf,1);
            all_acc_subjects = zeros(num_timepoints, numel(subjects));
            all_perm_subjects = zeros(num_timepoints, nPerm, numel(subjects));
        end
        all_acc_subjects(:,s) = perf;

%         %% --- PERMUTATIONS (PARALLEL) ---
%         if permutation == 1
%             perm_metrics = zeros(size(perf,1), nPerm);
%             parfor p = 1:nPerm
%                 y_perm = y(randperm(length(y)));
%                 [perf_perm, ~] = mv_classify_across_time(cfg, cue_hfa_ds, y_perm);
%                 perm_metrics(:,p) = perf_perm;
%             end
%             all_perm_subjects(:,:,s) = perm_metrics;
%         end
      
    end

    %% --- REMOVE SUBJECTS WITH ALL-ZERO ACCURACY ---
    subjects_to_keep = any(all_acc_subjects ~= 0, 1);
    all_acc_subjects = all_acc_subjects(:,subjects_to_keep);

    if permutation == 1
        all_perm_subjects = all_perm_subjects(:, :, subjects_to_keep);

        %% --- ZSCORE AGAINST PERMUTATIONS (VECTORIZE) ---
        mu_perm = mean(all_perm_subjects,2);
        sigma_perm = std(all_perm_subjects,0,2);
        zscore_acc = (all_acc_subjects - squeeze(mu_perm)) ./ squeeze(sigma_perm);
    
        % Average across subjects
        mean_z = mean(zscore_acc,2);
        sem_z  = std(zscore_acc,0,2)/sqrt(size(zscore_acc,2));
    
        %% --- PLOT ---
        figure; hold on;
        time_vec = cue_tfr.time(1:downsample_factor:end);
    
        fill([time_vec fliplr(time_vec)], [mean_z+sem_z; flipud(mean_z-sem_z)]', ...
             [0.7 0.7 1], 'FaceAlpha',0.3,'EdgeColor','none');
        plot(time_vec, mean_z, 'b', 'LineWidth', 2);
    
        xlabel('Time (s)');
        ylabel('Z-scored decoding');
        title('Decoding z-scored against permutations');
    else
        %% Statistics
        time_vec = cue_tfr.time(1:downsample_factor:end);
        stats_array = all_acc_subjects - chancelevel;
        nTime = size(stats_array,1);
        nSubj = size(stats_array,2);

        % Compute observed t-statistic at each timepoint
        obs_t = mean(stats_array, 2) ./ (std(stats_array, 0, 2)/sqrt(nSubj));

        % Permutation loop
        perm_max_cluster = [];

        for p = 1:nPerm
            signs = randi([0 1], nSubj,1)*2 - 1; % randomly flip sign for each subject
            perm_data = stats_array .* signs';
            t_perm = mean(perm_data,2) ./ (std(perm_data,0,2)/sqrt(nSubj));
            % find clusters of consecutive points above threshold
%             thresh = tinv(1 - alpha, nSubj - 1);   % one-tailed
            clust = bwconncomp(t_perm > thresh);
            
            % keep only clusters >= min_cluster_pts
            validClusters = cellfun(@numel, clust.PixelIdxList) >= min_cluster_pts;
            clust.PixelIdxList = clust.PixelIdxList(validClusters);

            cluster_sums = zeros(1,length(clust.PixelIdxList));
            for c = 1:length(clust.PixelIdxList)
                cluster_sums(c) = sum(t_perm(clust.PixelIdxList{c}));
            end
            perm_max_cluster = [perm_max_cluster, cluster_sums];
        end

        % Compute clusters in observed data
%         thresh = tinv(1 - alpha, nSubj - 1);   % one-tailed
        obs_clust = bwconncomp(obs_t > thresh);

        % keep only clusters >= min_cluster_pts
        validClusters = cellfun(@numel, obs_clust.PixelIdxList) >= min_cluster_pts;
        obs_clust.PixelIdxList = obs_clust.PixelIdxList(validClusters);

        obs_cluster_sums = zeros(1,length(obs_clust.PixelIdxList));
        for c = 1:length(obs_clust.PixelIdxList)
            obs_cluster_sums(c) = sum(obs_t(obs_clust.PixelIdxList{c}));
        end

        % Compute p-values for each observed cluster
        p_values = zeros(1,length(obs_cluster_sums));
        for c = 1:length(obs_cluster_sums)
            p_values(c) = mean(perm_max_cluster >= obs_cluster_sums(c));
        end
        
        % Display clusters and p-values
        disp('Observed clusters and p-values:');
        for c = 1:length(obs_cluster_sums)
            fprintf('Cluster %d: timepoints %s, p = %.4f\n', ...
            c, mat2str(obs_clust.PixelIdxList{c}), p_values(c));
        end
        
        % --- Plot t-statistic with significant clusters highlighted ---
        figure; hold on;
        plot(time_vec, obs_t, 'b', 'LineWidth', 2);
        ylim([-3, 6])
        xlabel('Time (s)'); 
        ylabel('t-value');
        title(sprintf('Memory Response Decoding: %s', roi));
        
        % Highlight significant clusters
        for c = 1:length(obs_clust.PixelIdxList)
            if p_values(c) < alpha
                cluster_times = time_vec(obs_clust.PixelIdxList{c});
                yLimits = ylim;
                patch([cluster_times(1) cluster_times(end) cluster_times(end) cluster_times(1)], ...
                [yLimits(1) yLimits(1) yLimits(2) yLimits(2)], ...
                'red', 'FaceAlpha',0.3, 'EdgeColor','none');
            end
        end
        plot(time_vec, obs_t, 'b', 'LineWidth', 2); % redraw line on top
        set(gca,'FontSize',10,'FontName','Arial');
        f = gcf;
        exportgraphics(f,fullfile(output_dir,[roi '_memory_response.pdf']),'Resolution',300)
    end
    
    %% --- SAVE DETAILED STATISTICAL REPORT ---
    txt_filename = fullfile(output_dir, [roi '_memory_response_stats.txt']);
    
    fid = fopen(txt_filename, 'w');
    
    fprintf(fid, 'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
    fprintf(fid, '======================================================\n\n');
    
    %% --- GENERAL INFORMATION ---
    fprintf(fid, 'ROI: %s\n', roi);
    fprintf(fid, 'Dependent variable: %s\n', dependent_var);
    fprintf(fid, 'Number of subjects: %d\n', sum(subjects_to_keep));
    fprintf(fid, 'Smoothing window: %d ms\n', win_ms);
    fprintf(fid, 'Downsample factor: %d\n', downsample_factor);
    fprintf(fid, 'Time resolution: %.6f s\n\n', mean(diff(time_vec)));
    
    %% --- DECODING CONFIGURATION ---
    fprintf(fid, 'Classifier: %s\n', cfg.classifier);
    fprintf(fid, 'Metric: %s\n', cfg.metric);
    fprintf(fid, 'Prior: %s\n', cfg.prior);
    fprintf(fid, 'Cross-validation: %s\n', cfg.cv);
    fprintf(fid, 'K-folds: %d\n', cfg.k);
    fprintf(fid, 'Repeats: %d\n', cfg.repeat);
    fprintf(fid, 'HFA frequency range: 70-201 Hz\n\n');
    
    %% --- STATISTICAL TEST DETAILS ---
    fprintf(fid, 'Statistical test: sign-flip permutation\n');
    fprintf(fid, 'Test statistic: t-test\n');
    fprintf(fid, 'Multiple comparison correction: cluster\n');
    fprintf(fid, 'Alpha level: %.4f\n', alpha);
    fprintf(fid, 'Number of permutations: %d\n\n', nPerm);
    
    fprintf(fid, 'Group-level design: within\n');
    fprintf(fid, 'Statistic: t-test\n');
    fprintf(fid, 'Tail: 1\n');
    fprintf(fid, 'Null hypothesis value: %.4f\n', chancelevel);
    fprintf(fid, 'Permutation test: sign-flip\n');
    fprintf(fid, 'Cluster statistic: maxsum\n');
    fprintf(fid, 'Cluster-forming threshold: %.4f\n', thresh);
    fprintf(fid, 'Minimum cluster size (timepoints): %d\n\n', min_cluster_pts);
    
    %% --- CLUSTER RESULTS ---
    n_sig_clusters = sum(p_values < alpha);
    fprintf(fid, 'Number of significant clusters: %d\n\n', n_sig_clusters);
    
    if n_sig_clusters == 0
        fprintf(fid, 'No significant clusters detected.\n');
    else
        for c = 1:length(obs_clust.PixelIdxList)
            if p_values(c) < alpha
                t_start = time_vec(obs_clust.PixelIdxList{c}(1));
                t_end = time_vec(obs_clust.PixelIdxList{c}(end));
                
                fprintf(fid, 'Cluster %d\n', c);
                fprintf(fid, '  Time range: %.4f – %.4f s\n', t_start, t_end);
                fprintf(fid, '  Cluster-level p-value: %.6f\n', p_values(c));
                fprintf(fid, '\n');
            end
        end
    end
    
    fclose(fid);
    fprintf('Saved statistical summary:\n%s\n', txt_filename);
end