%% Rewritten MVPA pipeline implementing MUST FIX items + electrode counts table
% - Stores permutations as [time x subject x nPerm]
% - Uses subject-level permutations to build group-level null by sampling
%   one perm map per subject for each group permutation.
% - Performs cluster-mass correction using t-statistic against chance level.
% - No z-scoring of accuracies.
% - Plots a table of ACC electrode counts per subject.
%
% NOTE: Assumes mv_classify_across_time and mv_regress behave as before.
%       Adjust paths and function names as needed.

clearvars; close all;

%% --- CONFIG ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback_fixed/');
if ~exist(output_dir,'dir'), mkdir(output_dir); end

ROIs = {'ACC'};  % list of ROIs
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

dependent_vars = {'rpe_category'};  % you can expand this list
downsample_factor = 5;
win_ms = 200;
do_permutation = true;
nPerm = 100;       % group-level permutations (>=1000 recommended)
nPerm_subject = 100; % permutations run per subject (can be equal or higher than group nPerm)
alpha = 0.05;       % cluster-level alpha

%% --- LOAD TABLES ONCE ---
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% Loop over dependent variables
for dv_i = 1:numel(dependent_vars)
    dependent_var = dependent_vars{dv_i};
    fprintf('=== DEP VAR: %s ===\n', dependent_var);
    
    % Containers that will be filled with varying #subjects (only keep subjects
    % with data)
    all_true_perf = [];           % time x nSubjects
    all_perm_perf = [];           % time x nSubjects x nPerm_subject
    subj_list_kept = {};
    subj_elec_counts = zeros(1, numel(subjects));
    time_vec = [];                % time vector (after downsample)
    chancelevel = [];             % set per DV below
    
    subj_counter = 0;
    % --- SUBJECT LOOP ---
    for s = 1:numel(subjects)
        subj = subjects{s};
        fprintf(' Subject %s ...\n', subj);
        
        % Select subject rows
        subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
        subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
        subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);
        subj_combined_df.phit = double(strcmp(subj_combined_df.hits, 'True'));
        
        % Skip if no learn_df rows
        if isempty(subj_learn_df)
            warning('No learn_df for %s', subj); continue;
        end
        
        % Tercile rpe categories within-subject
        if any(strcmp(subj_learn_df.Properties.VariableNames,'rpe'))
            edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
            subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative','neutral','positive'});
        end
        if any(strcmp(subj_combined_df.Properties.VariableNames,'rpe'))
            edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
            subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative','neutral','positive'});
        end
        
        % Choose encoding trials and RT mask
        if strcmp(dependent_var,'phit')
            encoding_trials = subj_combined_df.trials_gamble;
            mask_rt = subj_combined_df.gamble_rt_gamble >= 0.3;
            encoding_trials = encoding_trials(mask_rt);
        else
            encoding_trials = subj_learn_df.trials;
            mask_rt = subj_learn_df.gamble_rt >= 0.3;
            encoding_trials = encoding_trials(mask_rt);
        end
        if isempty(encoding_trials)
            warning('No encoding trials for %s', subj); continue;
        end
        
        % Load electrode info and count ROI electrodes
        elec_df_path = fullfile(base_dir, 'projects','guLab','Salman','EphysAnalyses', subj, 'Day1_reref_elec_df');
        if exist(elec_df_path,'file')
            elec_df = readtable(elec_df_path);
        else
            warning('Electrode df not found for %s at %s', subj, elec_df_path);
            elec_df = table(); % proceed but will likely fail later
        end
        roi = ROIs{1};  % only ACC in current list
        if ismember('salman_region', elec_df.Properties.VariableNames)
            roi_electrodes = elec_df.label(strcmp(elec_df.salman_region, roi));
        else
            roi_electrodes = {};
        end
        subj_elec_counts(s) = numel(roi_electrodes); %#ok<SAGROW>
        
        if isempty(roi_electrodes)
            warning('No electrodes in ROI for %s', subj); continue;
        end
        
        % Load feedback TFR (assumes same variable names as your original load)
        filepath = fullfile(base_dir,'projects','guLab','Salman','EphysAnalyses',subj,'scratch','TFR');
        if ~exist(fullfile(filepath,'feedback_start-tfr.mat'),'file')
            warning('TFR file missing for %s', subj); continue;
        end
        fb = load(fullfile(filepath, 'feedback_start-tfr.mat'));
        % Map into consistent struct
        feedback_tfr.powspctrm = fb.powspctrm; % trials x chans x freq x time
        feedback_tfr.freq = fb.freqs;
        feedback_tfr.time = fb.times;
        feedback_tfr.label = fb.ch_names;
        feedback_tfr.fsample = fb.sfreq;
        
        % Select ROI electrodes in TFR
        [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
        idx_fb(idx_fb==0) = []; % remove not found
        if isempty(idx_fb)
            warning('No ROI channels found in TFR for %s', subj); continue;
        end
        feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
        
        % Extract HFA 70-200 Hz (note: use <=200)
        hfa_mask = feedback_tfr.freq >= 70 & feedback_tfr.freq <= 200;
        feedback_hfa = squeeze(mean(feedback_tfr.powspctrm(:,: , hfa_mask, :), 3)); % trials x elec x time
        if ndims(feedback_hfa) == 2
            feedback_hfa = reshape(feedback_hfa, [size(feedback_hfa,1), 1, size(feedback_hfa,2)]);
        end
        
        % Smooth
        fs = feedback_tfr.fsample;
        win_size = round(fs * (win_ms/1000));
        if win_size < 1, win_size = 1; end
        feedback_hfa_smooth = movmean(feedback_hfa, win_size, 3);
        
        % Select encoding trials and downsample
        feedback_hfa_smooth = feedback_hfa_smooth(encoding_trials,:,:);
        feedback_hfa_ds = feedback_hfa_smooth(:,:,1:downsample_factor:end); % trials x elec x time
        nTime = size(feedback_hfa_ds,3);
        if isempty(time_vec)
            time_vec = feedback_tfr.time(1:downsample_factor:end);
        end
        
        % Define labels y and chancelevel
        switch dependent_var
            case 'reward'
                y_full = subj_learn_df.reward;
                y = y_full(mask_rt);
                chance = 0.5;
            case 'rpe'
                y_full = subj_learn_df.rpe;
                y = y_full(mask_rt);
                chance = 0;
            case 'phit'
                y_full = subj_combined_df.phit;
                y = y_full(mask_rt);
                chance = 0.5;
            case 'rpe_category'
                y_full = subj_learn_df.rpe_category;
                y = y_full(mask_rt);
                chance = 1/3;
            otherwise
                error('Unknown dependent_var: %s', dependent_var);
        end
        chancelevel = chance; %#ok<NASGU>
        % Convert categorical to numeric labels if needed
        if iscategorical(y) || iscellstr(y)
            [~, ~, ynum] = unique(y);
            y = double(ynum);
        else
            y = double(y);
        end
        
        % Configure MVPA-light style cfg
        cfg = [];
        switch dependent_var
            case {'reward','phit'}
                cfg.classifier = 'lda';
                cfg.metric = 'acc';
                cfg.prior = 'uniform';
            case 'rpe_category'
                cfg.classifier = 'multiclass_lda';
                cfg.metric = 'acc'; 
                cfg.prior = 'uniform';
                cfg.reg = 'shrinkage';
                cfg.lambda = 'auto';
            case 'rpe'
                cfg.regressor = 'ridge';
                cfg.metric = 'r_squared';
                cfg.lambda = 'auto';
                cfg.generalization = 'time';
        end
        cfg.cv = 'leaveout'; 
%         cfg.k = 5; 
%         cfg.repeat = 10; 
        cfg.feedback = 0;
        
        % Run true decoding (time-resolved)
        if strcmp(dependent_var,'rpe')
            [perf_true, res_true] = mv_regress(cfg, feedback_hfa_ds, y);
        else
            [perf_true, res_true] = mv_classify_across_time(cfg, feedback_hfa_ds, y);
        end
        % perf_true: nTime x 1 (or nTime x metrics). We assume nTime x 1.
        
        % Run subject-level permutations (store time x nPerm_subject)
        if do_permutation
            perf_perm_sub = zeros(nTime, nPerm_subject);
            rng('shuffle'); % different random seeds per subject
            parfor p = 1:nPerm_subject
                y_perm = y(randperm(length(y)));
                if strcmp(dependent_var,'rpe')
                    perf_p = mv_regress(cfg, feedback_hfa_ds, y_perm);
                else
                    perf_p = mv_classify_across_time(cfg, feedback_hfa_ds, y_perm);
                end
                perf_perm_sub(:,p) = perf_p(:);
            end
        else
            perf_perm_sub = [];
        end
        
        % Keep subject results
        subj_counter = subj_counter + 1;
        subj_list_kept{subj_counter} = subj; %#ok<SAGROW>
        all_true_perf(:,subj_counter) = perf_true(:); %#ok<SAGROW>
        if do_permutation
            % store as time x subj x perm
            if isempty(all_perm_perf)
                all_perm_perf = zeros(nTime, numel(subjects), nPerm_subject);
            end
            all_perm_perf(:, subj_counter, :) = perf_perm_sub;
        end
        
        % Save per-subject result struct (optional)
        subj_results{subj_counter}.subj = subj;
        subj_results{subj_counter}.perf = perf_true(:);
        subj_results{subj_counter}.perm = perf_perm_sub; % time x nPerm_subject
        subj_results{subj_counter}.n_elec = numel(idx_fb);
        
    end % subject loop
    
    % Trim subj_elec_counts to actual subject list
    elec_count_table = table(subjects(:), subj_elec_counts(:), 'VariableNames', {'subject','n_acc_electrodes'});
    % Show only kept subjects
    kept_mask = ismember(elec_count_table.subject, subj_list_kept);
    elec_count_table_kept = elec_count_table(kept_mask,:);
    
    % Plot electrode counts table as bar + uitable
    figure('Name','ACC electrode counts per subject','NumberTitle','off');
    subplot(2,1,1);
    bar(elec_count_table_kept.n_acc_electrodes);
    set(gca,'XTick',1:height(elec_count_table_kept),'XTickLabel',elec_count_table_kept.subject,'XTickLabelRotation',45);
    ylabel('# ACC electrodes');
    title('ACC electrode counts (kept subjects)');
    subplot(2,1,2);
    uit = uitable('Data', table2cell(elec_count_table_kept), 'ColumnName', elec_count_table_kept.Properties.VariableNames, ...
        'Units','Normalized','Position',[0 0 1 0.45]);
    drawnow;
    savefig(fullfile(output_dir, sprintf('%s_ACC_electrode_counts.fig', dependent_var)));
    
    % If no subjects survived, skip stats
    if isempty(all_true_perf)
        warning('No valid subjects for %s, skipping group stats.', dependent_var);
        continue;
    end
    
    % --- GROUP-LEVEL CLUSTER PERMUTATION TEST ---
    fprintf('Running group-level cluster permutation test...\n');
    [nTime, nSubj] = size(all_true_perf);
    % Observed one-sample t-statistic against chancelevel:
    % t = mean(diff)/ (std(diff)/sqrt(n))
    diff_obs = all_true_perf - chancelevel;
    mean_obs = mean(diff_obs,2);
    std_obs = std(diff_obs,0,2);
    t_obs = mean_obs ./ (std_obs ./ sqrt(nSubj));
    % set tcrit for one-sided test (positive direction)
    df = nSubj - 1;
    tcrit = tinv(1 - alpha, df);  % one-sided alpha
    
    % Find clusters in observed t_obs (positive clusters only)
    thr_mask_obs = t_obs > tcrit;
    clusters_obs = bwconncomp(thr_mask_obs);
    cluster_stats_obs = [];
    for c = 1:clusters_obs.NumObjects
        idxs = clusters_obs.PixelIdxList{c};
        cluster_stats_obs(c).indices = idxs;
        cluster_stats_obs(c).mass = sum(t_obs(idxs)); % cluster-mass = sum of t-values
        cluster_stats_obs(c).size = numel(idxs);
    end
    
    % Build group-level null distribution:
    % For each group permutation g: sample for each subject s a random perm index
    % from that subject's subject-level permutations and compute t-statistic.
    max_cluster_mass_null = zeros(nPerm,1);
    rng('shuffle');
    % Confirm all_perm_perf is filled for kept subjects
    % Perm shape: nTime x nSubjectsPossible x nPerm_subject
    % We use columns corresponding to subj_counter order stored earlier (1:subj_counter)
    subj_count_actual = size(all_true_perf,2);
    for g = 1:nPerm
        % build matrix time x subjects for this group perm by sampling one perm per subject
        perm_matrix = zeros(nTime, subj_count_actual);
        for ss = 1:subj_count_actual
            % sample a random permutation index for subject ss
            idx_rand = randi(nPerm_subject);
            perm_matrix(:,ss) = all_perm_perf(:, ss, idx_rand);
        end
        % compute t-statistic across subjects comparing perm_matrix to chancelevel
        diff_perm = perm_matrix - chancelevel;
        mean_perm = mean(diff_perm,2);
        std_perm = std(diff_perm,0,2);
        % To avoid division by zero, set std small values to eps
        std_perm(std_perm==0) = eps;
        t_perm = mean_perm ./ (std_perm ./ sqrt(nSubj));
        % threshold and compute max cluster mass in this perm
        thr_mask_perm = t_perm > tcrit;
        clusters_perm = bwconncomp(thr_mask_perm);
        max_mass = 0;
        for c = 1:clusters_perm.NumObjects
            idxs = clusters_perm.PixelIdxList{c};
            mass = sum(t_perm(idxs));
            if mass > max_mass, max_mass = mass; end
        end
        max_cluster_mass_null(g) = max_mass;
    end
    
    % Now evaluate observed clusters against null distribution
    % Null is distribution of max cluster mass across time (one-sided)
    cluster_pvals = [];
    for c = 1:numel(cluster_stats_obs)
        observed_mass = cluster_stats_obs(c).mass;
        % p = proportion of null max masses >= observed_mass
        pval = mean(max_cluster_mass_null >= observed_mass);
        cluster_stats_obs(c).pval = pval;
    end
    
    % Create a mask of significant clusters (p < alpha)
    sig_mask = false(nTime,1);
    for c = 1:numel(cluster_stats_obs)
        if cluster_stats_obs(c).pval < alpha
            sig_mask(cluster_stats_obs(c).indices) = true;
        end
    end
    
    % Save results struct
    group_results.dependent_var = dependent_var;
    group_results.time = time_vec;
    group_results.all_true_perf = all_true_perf;
    group_results.all_perm_perf = all_perm_perf(:,1:subj_count_actual,:); % trim
    group_results.t_obs = t_obs;
    group_results.tcrit = tcrit;
    group_results.sig_mask = sig_mask;
    group_results.cluster_stats_obs = cluster_stats_obs;
    save(fullfile(output_dir, sprintf('group_results_%s.mat', dependent_var)), 'group_results');
    
    % Plot mean accuracy across subjects with significant clusters highlighted
    mean_acc = mean(all_true_perf,2);
    sem_acc = std(all_true_perf,0,2) / sqrt(size(all_true_perf,2));
    fig = figure('Name',sprintf('Decoding %s',dependent_var),'NumberTitle','off','Visible','on');
    hold on;
    % shaded error
    fill([time_vec fliplr(time_vec)], [mean_acc+sem_acc; flipud(mean_acc-sem_acc)]', ...
         [0.9 0.9 0.9],'EdgeColor','none');
    plot(time_vec, mean_acc, 'k', 'LineWidth', 2);
    % mark significant cluster regions
    if any(sig_mask)
        ylims = ylim;
        % draw semi-transparent rectangles for clusters
        d = diff([0; sig_mask; 0]);
        starts = find(d==1);
        ends = find(d==-1)-1;
        for cc = 1:numel(starts)
            x1 = time_vec(starts(cc));
            x2 = time_vec(ends(cc));
            rectangle('Position',[x1, ylims(1), x2-x1, ylims(2)-ylims(1)], 'FaceColor',[0.6 0.85 0.6], 'EdgeColor','none','FaceAlpha',0.3);
        end
    end
    xlabel('Time (s)'); ylabel(sprintf('Mean %s', cfg.metric));
    title(sprintf('%s decoding (n=%d subjects)', dependent_var, size(all_true_perf,2)));
    saveas(fig, fullfile(output_dir, sprintf('%s_mean_decoding_with_clusters.png', dependent_var)));
    
    % Print cluster summary
    fprintf('Clusters (observed):\n');
    for c = 1:numel(cluster_stats_obs)
        fprintf(' cluster %d: size=%d, mass=%.3f, p=%.4f\n', c, cluster_stats_obs(c).size, cluster_stats_obs(c).mass, cluster_stats_obs(c).pval);
    end
    
    % Also save electrode count table
    writetable(elec_count_table_kept, fullfile(output_dir, sprintf('%s_ACC_electrode_counts.csv', dependent_var)));
    
end % dependent var loop

fprintf('Done. Results saved to %s\n', output_dir);
