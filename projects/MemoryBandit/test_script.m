%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/crossdecoding/');

ROIs = {'HPC', 'AMY'};  %  'dmPFC', 'dlPFC', 'ACC', 'HPC', 'AMY'
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035',     'MS036', 'UI006', 'UI007'}; % 

dependent_var = 'phit';
downsample_factor = 5;    % match your Python version
win_ms = 200;             % smoothing window
nPerm = 200;             % for cluster permutation test
alpha = 0.05;
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- LOOP OVER ROIs ---
for r = 1:numel(ROIs)
    roi = ROIs{r};
    fprintf('Processing ROI: %s\n', roi);
    
    all_acc_subjects = [];  % store matrices for all subjects
    all_results_subjects = cell(1, numel(subjects));  % store results struct

    %% --- LOOP OVER SUBJECTS ---
    for s = 1:numel(subjects)
        subj = subjects{s};
        fprintf('  Subject: %s\n', subj);
       
        %% --- SELECT SUBJECT DATA ---
        subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
        subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
        subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);


        %% Process RPE categories using tercile split within subject
        edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
        subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});
        edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
        subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative', 'neutral', 'positive'});
        
        subj_combined_df.phit = strcmp(subj_combined_df.hits, 'True');
        %% Trial mapping and remove fast RTs

        encoding_trials = subj_combined_df.trials_gamble; % order of encoding trials
        retrieval_trials = subj_combined_df.trials_mem;

%         mask_rt_enc = subj_combined_df.gamble_rt_gamble >= 0.3;
        mask_rt_retr = subj_combined_df.recog_rt >= 0.3;
            
        encoding_trials = encoding_trials(mask_rt_retr);
        retrieval_trials = retrieval_trials(mask_rt_retr);

        
        %% Load electrode info
        elec_df = readtable(fullfile(base_dir, 'projects','guLab','Salman','EphysAnalyses', subj, 'Day1_reref_elec_df'));
        roi_electrodes = elec_df.label(strcmp(elec_df.salman_region, roi));
        if isempty(roi_electrodes)
            warning('No electrodes found for %s in %s', subj, roi);
            continue;
        end
        
        %% Load TFR data
        filepath = fullfile(base_dir,'projects','guLab','Salman','EphysAnalyses',subj,'scratch','TFR');
        fb = load(fullfile(filepath, 'feedback_start-tfr.mat'));
        cue = load(fullfile(filepath, 'cue_on-tfr.mat'));
        
        feedback_tfr.powspctrm = fb.powspctrm;
        feedback_tfr.freq = fb.freqs;
        feedback_tfr.time = fb.times;
        feedback_tfr.label = fb.ch_names;
        feedback_tfr.fsample = fb.sfreq;
        
        cue_tfr.powspctrm = cue.powspctrm;
        cue_tfr.freq = cue.freqs;
        cue_tfr.time = cue.times;
        cue_tfr.label = cue.ch_names;
        cue_tfr.fsample = cue.sfreq;
        
        %% Select ROI electrodes
        [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
        [~, idx_cue] = ismember(roi_electrodes, cue_tfr.label);
        feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
        cue_tfr.powspctrm = cue_tfr.powspctrm(:, idx_cue, :, :);
        
        %% Extract HFA 70–200 Hz
        hfa_mask = feedback_tfr.freq >= 70 & feedback_tfr.freq <= 201;
        feedback_hfa = squeeze(mean(feedback_tfr.powspctrm(:, :, hfa_mask, :), 3));  % trials x electrodes x time
        cue_hfa = squeeze(mean(cue_tfr.powspctrm(:, :, hfa_mask, :), 3));
                
        Ndims = ndims(feedback_hfa); 
        if Ndims < 3 
            feedback_hfa = reshape(feedback_hfa, [size(feedback_hfa, 1), 1, size(feedback_hfa, 2)]); 
        end

        Ndims = ndims(cue_hfa); 
        if Ndims < 3 
            cue_hfa = reshape(cue_hfa, [size(cue_hfa, 1), 1, size(cue_hfa, 2)]); 
        end
%         feedback_hfa = permute(feedback_hfa, [3 1 2]);  
%         cue_hfa = permute(cue_hfa, [3 1 2]);
        
        %% --- SMOOTH (VECTORIZE) ---
        fs = feedback_tfr.fsample;
        win_size = round(fs * (win_ms/1000));
        feedback_hfa_smooth = movmean(feedback_hfa, win_size, 3);  % vectorized smoothing
        cue_hfa_smooth = movmean(cue_hfa, win_size, 3);  % vectorized smoothing

        
        %% Select encoding trials
        feedback_hfa_smooth = feedback_hfa_smooth(encoding_trials,:,:);
        cue_hfa_smooth = cue_hfa_smooth(retrieval_trials,:,:);

        %% Downsample
        feedback_hfa_ds = feedback_hfa_smooth(:,:,1:downsample_factor:end);
        cue_hfa_ds = cue_hfa_smooth(:,:,1:downsample_factor:end);

        %% Define labels
        switch dependent_var
            case 'rpe_category'
                y = subj_combined_df.(dependent_var);
                y = y(mask_rt_retr);
                chancelevel = 0.33;
            case 'rpe'
                y = subj_combined_df.(dependent_var);
                y = y(mask_rt_retr);
                chancelevel = 0;
            case 'phit'
                y = subj_combined_df.(dependent_var);
                y = y(mask_rt_retr);
                chancelevel = 0.5;
        end
        
        y_train = double(y);
        
        %% --- MVPA-LIGHT CONFIGURATION ---
        cfg = [];
        switch dependent_var
            case 'reward'
                cfg.classifier = 'lda'; cfg.metric = 'acc';
            case 'phit'
                cfg.classifier = 'lda'; cfg.metric = 'acc';
            case 'rpe_category'
                cfg.classifier = 'multiclass_lda'; cfg.metric = 'acc'; cfg.prior = 'uniform';
            case 'rpe'
                cfg.regressor = 'ridge'; cfg.metric = 'r_squared';
                cfg.hyperparameter = 'lambda';
                cfg.lambda = logspace(-4,4,9);
                cfg.lambda_select = 'nested';
                cfg.generalization = 'time';
        end
        cfg.cv = 'kfold'; cfg.k = 5; cfg.repeat = 20; cfg.feedback = 0;

%% Run it 
        if strcmp(dependent_var, 'rpe')
            [nTrials_fb, nElec, nTimes_fb] = size(feedback_hfa_ds);
            [~, ~, nTimes_cue] = size(cue_hfa_ds);
            
            pred_matrix = zeros(nTimes_fb, nTimes_cue);  % store correlation
            
            for t_fb = 1:nTimes_fb
                % training data at one feedback timepoint
                Xtrain = squeeze(feedback_hfa_ds(:, :, t_fb));  % [nTrials_fb x nElec]
                
                for t_cue = 1:nTimes_cue
                    Xtest = squeeze(cue_hfa_ds(:, :, t_cue));  % [nTrials_cue x nElec]                    
                    % run transfer regression
                    result = mv_regress(cfg, Xtrain, y_train, Xtest, y_train);  
                    % result is a struct with field 'pred' (predicted y) and 'perf' (metric)
                    
                    pred_matrix(t_fb, t_cue) = result;  % correlation r
                end
            end
            all_acc_subjects(:,:,s) = pred_matrix;
        else
            [nTrials, nElec, nTimes_fb] = size(feedback_hfa_ds);
            [~, ~, nTimes_cue] = size(cue_hfa_ds);
    
            %% OPTION 1: Use mv_classify_timextime (fastest if available)
            % Train on all feedback timepoints, test on all cue timepoints
            cfg_cross = cfg;
            cfg_cross.generalization = 'cross';  % or check mvpa-light docs
            
            % This should do the full cross-temporal matrix in one call
            [acc_matrix, result] = mv_classify_timextime(cfg, feedback_hfa_ds, y_train, cue_hfa_ds, y_train);

            all_acc_subjects(:,:,s) = acc_matrix;
            all_results_subjects{s} = result;
        end
    end

    % Find subjects that are all zeros
    subjects_to_keep = squeeze(any(any(all_acc_subjects,1),2));  
    % subjects_to_keep is a logical vector, 1 = keep, 0 = drop
    
    % Filter the matrix
    all_acc_subjects = all_acc_subjects(:,:,subjects_to_keep);
    all_results_subjects = all_results_subjects(subjects_to_keep);

    result_average_patients = mv_combine_results(all_results_subjects, 'average');
    
    %% --- INPUT ---
    cfg_stat = [];
    cfg_stat.metric          =  cfg.metric;
    cfg_stat.test            = 'permutation';
    cfg_stat.correctm        = 'cluster';  % correction method is cluster
    cfg_stat.n_permutations  = 2000;
    % Clusterstatistic is the actual statistic used for the clustertest.
    % Normally the default value 'maxum' is used, we are setting it here
    % explicitly for clarity. Maxsum adds up the statistics calculated at each
    % time point (the latter are set below using cfg_stat.statistic)
    cfg_stat.clusterstatistic = 'maxsum';
    cfg_stat.alpha           = 0.05; % use standard significance threshold of 5%
    % Level 2 stats design: we have to choose between within-subject and
    % between-subjects. Between-subjects is relevant when there is two
    % different experimental groups (eg patients vs controls) and we want to
    % investigate whether their MVPA results are significantly different. Here,
    % we have only one group and we want to see whether the AUC is
    % significantly different from a null value, hence the statistical design
    % is within-subject
    cfg_stat.design          = 'within';
    % cfg_stat.statistic defines how the difference between the AUC values and
    % the null is calculated at each time point (across subjects).
    % We can choose t-test or its nonparametric counterpart Wilcoxon test. We
    % choose Wilcoxon here.
    cfg_stat.statistic       = 'ttest';
    % The null value for AUC (corresponding to a random classifier) is 0.5
    cfg_stat.null            = chancelevel;
    % clustercritval is a parameter that needs to be selected by the user. In a
    % Level 2 (group level) test, it represents the critical cutoff value for
    % the statistic. Here, we selected Wilcoxon, so clustercritval corresponds
    % to the cutoff value for the z-statistic which is obtained by a normal
    % approximation.
    cfg_stat.clustercritval  = 1.7;
    cfg_stat.tail = 1;
    % z-val = 1.65 corresponds to uncorrected p-value = 0.1
    % z-val = 1.96 corresponds to uncorrected p-value = 0.05
    % z-val = 2.58 corresponds to uncorrected p-value = 0.01
    % Note that these z-values correspond to a two-sided test. In many cases it
    % suffices to use a one-sided test. In this case, the following cutoff
    % values are obtained:
    % z-val = 1.282 corresponds to uncorrected p-value = 0.1
    % z-val = 1.645 corresponds to uncorrected p-value = 0.05
    % z-val = 2.326 corresponds to uncorrected p-value = 0.01
    stat_level2 = mv_statistics(cfg_stat, all_results_subjects);
    
    % Get downsampled time vectors to match matrix dimensions
    time_fb_ds = feedback_tfr.time(1:downsample_factor:end);
    time_cue_ds = cue_tfr.time(1:downsample_factor:end);
    
%     % Verify dimensions match
%     assert(length(time_fb_ds) == size(tvals, 1), 'Feedback time mismatch');
%     assert(length(time_cue_ds) == size(tvals, 2), 'Cue time mismatch');
%     
    % Create figure with multiple panels
    figure('Position', [100 100 1200 400]);
    
    % Panel 1: Mean accuracy
    subplot(1,3,1);
%     imagesc(time_cue_ds, time_fb_ds, tvals);
    mv_plot_2D(result_average_patients.perf{1, 1}  , 'x',time_cue_ds, 'y',time_fb_ds)
    axis xy;
    xlabel('Retrieval time (s)');
    ylabel('Encoding time (s)');
    title('Mean Accuracy (all)');
    colorbar;
    caxis([chancelevel chancelevel+0.1]);  % scale colorbar

    % Panel 2: Wilcoxon z
    subplot(1,3,2);
%     imagesc(time_cue_ds, time_fb_ds, mean_data + 0.33);  % add chance back
    mv_plot_2D(stat_level2.statistic, 'x',time_cue_ds, 'y',time_fb_ds,...
    'colorbar_title', cfg_stat.statistic)
    axis xy;
    xlabel('Retrieval time (s)');
    ylabel('Encoding time (s)');
    title('Wilcoxon Z');
    colorbar;
    caxis([-3 3]);  % scale colorbar
    
    % Panel 3: Significant clusters only
    subplot(1,3,3);
    mv_plot_2D(stat_level2.statistic, 'x',time_cue_ds, 'y',time_fb_ds,...
     'mask', stat_level2.mask)
%     imagesc(time_cue_ds, time_fb_ds, tvals .* sig_mask_final);
%     hold on;
%     % Overlay cluster boundaries only if there are significant clusters
%     if any(sig_mask_final(:))
%         contour(time_cue_ds, time_fb_ds, sig_mask_final, 1, 'k', 'LineWidth', 2);
%     end
    axis xy;
    xlabel('Retrieval time (s)');
    ylabel('Encoding time (s)');
%     title(sprintf('Significant Clusters (n=%d)', length(sig_clusters)));
    colorbar;

    f = gcf;
    exportgraphics(f,fullfile(output_dir,[roi '_phit.pdf']),'Resolution',300)

end
% save(fullfile(output_dir, [roi '_all_acc_subjects.mat']), '-v7.3');

