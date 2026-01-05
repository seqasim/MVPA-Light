%% ============================================================
%  FEEDBACK → CUE CROSS-PERIOD DECODING (HFA, RPE Category)
%  Matches modernized feedback-period decoding pipeline
%  Author: Salman Qasim
% =============================================================

clear; clc;

%% --- FIX: Remove FieldTrip paths to avoid MEX signature issues ---
% FieldTrip's MEX files have code signature issues on macOS
% MATLAB's native statistics functions work fine for MVPA-Light
ft_base = '/Users/salmanqasim/Documents/MATLAB/plot_recon_roi/dependencies/fieldtrip-20240110';
ft_paths_to_remove = {
    fullfile(ft_base, 'external', 'stats');
    fullfile(ft_base, 'src');
    fullfile(ft_base, 'external', 'signal');  % Also remove signal to be safe
};

current_path = path;
for i = 1:numel(ft_paths_to_remove)
    if contains(current_path, ft_paths_to_remove{i})
        rmpath(ft_paths_to_remove{i});
        fprintf('Removed from path: %s\n', ft_paths_to_remove{i});
    end
end

%% --- CONFIGURATION ---
rng(42);  % Set seed for reproducibility

base_dir    = '/Volumes/T7 Shield';
load_dir    = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
ephys_dir   = fullfile(base_dir, 'projects', 'guLab', 'Salman', 'EphysAnalyses');
output_dir = '/Volumes/T7 Shield/scratch/MemoryBandit/MVPA/crossdecoding';

subjects = {'MS012','UI001','MS016','MS017','UI002','MS019','MS020','MS022','MS023',...
            'UI003','MS025','MS026','MS028','MS030','UI004','MS035','MS036','UI006','UI007'};

ROIs = {'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY'};        % can loop over multiple
downsample_factor = 5;
smooth_win_ms = 150;
alpha = 0.05;
nPerm = 5000;

%% --- PLOT AESTHETICS CONFIGURATION ---
plot_cfg = struct();
plot_cfg.font_name = 'Arial';
plot_cfg.font_size_label = 12;
plot_cfg.font_size_tick = 10;
plot_cfg.font_size_roi = 12;
plot_cfg.tick_dir = 'out';
plot_cfg.line_width = 1;
plot_cfg.contour_width = 2;
plot_cfg.contour_color = 'w';
plot_cfg.clim = [31 40];           % Color limits (accuracy in %)
plot_cfg.xticks = 0:0.2:1.0;       % X-axis tick positions
plot_cfg.yticks = 0:0.5:1.5;       % Y-axis tick positions
plot_cfg.tick_length = [0.025 0.025];
plot_cfg.colormap = 'parula';      % 'parula', 'viridis', 'jet', etc.

%% --- LOAD BEHAVIOR DATA ONCE (optimization) ---
fprintf('Loading behavioral data...\n');
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- STORAGE FOR ALL ROIs (for combined plotting) ---
all_acc_rois = cell(1, numel(ROIs));
sig_mask_rois = cell(1, numel(ROIs));
time_fb_rois = cell(1, numel(ROIs));
time_cue_rois = cell(1, numel(ROIs));
valid_rois = false(1, numel(ROIs));

%% --- LOOP OVER ROIs ---
for r = 1:numel(ROIs)
    roi = ROIs{r};
    fprintf('\n===== ROI: %s =====\n', roi);
    
    all_acc = [];
    subj_ids = {};  % Track which subjects have valid data for this ROI
    n_elec_subj = zeros(1, numel(subjects));  % Track electrode count per subject

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

        %% Trial mapping and remove fast RTs
        enc_trials = subj_combined_df.trials_gamble;
        ret_trials = subj_combined_df.trials_mem;
%         mask_rt = subj_combined_df.gamble_rt_gamble >= 0.3;
%         enc_trials = enc_trials(mask_rt);
%         ret_trials = ret_trials(mask_rt);

        %% --- LOAD ELECTRODES ---
        elec_df = readtable(fullfile(ephys_dir, subj, 'Day1_reref_elec_df'));
        roi_elec = elec_df.label(strcmp(elec_df.salman_region, roi));
        if isempty(roi_elec)
            warning('No %s electrodes for %s', roi, subj);
            continue;
        end
        n_elec_subj(s) = numel(roi_elec);  % Track electrode count

        %% --- LOAD TFR DATA ---
        tfr_dir = fullfile(ephys_dir, subj, 'scratch', 'TFR');
        fb = load(fullfile(tfr_dir, 'feedback_start-tfr.mat'));
        cue = load(fullfile(tfr_dir, 'cue_on-tfr.mat'));

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

        %% --- SELECT ROI ELECTRODES ---
        [~, idx_fb] = ismember(roi_elec, feedback_tfr.label);
        [~, idx_cue] = ismember(roi_elec, cue_tfr.label);
        feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
        cue_tfr.powspctrm = cue_tfr.powspctrm(:, idx_cue, :, :);

        %% --- COMPUTE HFA (70–200 Hz) ---
        hfa_mask = feedback_tfr.freq >= 70 & feedback_tfr.freq <= 201;
        feedback_hfa = squeeze(mean(feedback_tfr.powspctrm(:,:,hfa_mask,:),3));
        cue_hfa = squeeze(mean(cue_tfr.powspctrm(:,:,hfa_mask,:),3));

        %% --- SMOOTH HFA TIME SERIES ---
        fs = feedback_tfr.fsample;
        win_size = round(fs * smooth_win_ms / 1000);
        feedback_hfa = movmean(feedback_hfa, win_size, 3);
        cue_hfa = movmean(cue_hfa, win_size, 3);
        
        Ndims = ndims(feedback_hfa); 
        if Ndims < 3 
            feedback_hfa = reshape(feedback_hfa, [size(feedback_hfa, 1), 1, size(feedback_hfa, 2)]); 
            cue_hfa = reshape(cue_hfa, [size(cue_hfa, 1), 1, size(cue_hfa, 2)]); 
        end

        %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
        z_thresh = 10;
        
        % Process feedback HFA
        [nTrials_fb, nElec_fb, nTime_fb] = size(feedback_hfa);
        for trial = 1:nTrials_fb
            for elec = 1:nElec_fb
                trace = squeeze(feedback_hfa(trial, elec, :));
                bad_idx = abs(trace) > z_thresh;
                
                if any(bad_idx)
                    good_idx = find(~bad_idx);
                    bad_idx_list = find(bad_idx);
                    
                    if numel(good_idx) >= 2
                        trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                    else
                        trace(trace > z_thresh) = z_thresh;
                        trace(trace < -z_thresh) = -z_thresh;
                    end
                    
                    feedback_hfa(trial, elec, :) = trace;
                end
            end
        end
        
        % Process cue HFA
        [nTrials_cue, nElec_cue, nTime_cue] = size(cue_hfa);
        for trial = 1:nTrials_cue
            for elec = 1:nElec_cue
                trace = squeeze(cue_hfa(trial, elec, :));
                bad_idx = abs(trace) > z_thresh;
                
                if any(bad_idx)
                    good_idx = find(~bad_idx);
                    bad_idx_list = find(bad_idx);
                    
                    if numel(good_idx) >= 2
                        trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                    else
                        trace(trace > z_thresh) = z_thresh;
                        trace(trace < -z_thresh) = -z_thresh;
                    end
                    
                    cue_hfa(trial, elec, :) = trace;
                end
            end
        end


        %% --- DOWNSAMPLE ---
        feedback_hfa = feedback_hfa(enc_trials,:,1:downsample_factor:end);
        cue_hfa = cue_hfa(ret_trials,:,1:downsample_factor:end);
        time_fb = feedback_tfr.time(1:downsample_factor:end);
        time_cue = cue_tfr.time(1:downsample_factor:end);

        %% --- LABELS (RPE CATEGORY) ---
        y_train_cat = subj_learn_df.rpe_category(enc_trials);
        y_train = grp2idx(y_train_cat); % 1=neg, 2=neutral, 3=pos

        %% --- CROSS-DECODE (TRAIN FEEDBACK → TEST CUE) ---
        cfg = [];
        cfg.classifier = 'multiclass_lda'; 
        cfg.metric = 'acc'; 
        cfg.prior = 'uniform';
        cfg.reg = 'ridge';
        cfg.cv = 'none'; % I don't need CV here because I am testing on a different period.

        [nTrials, nElec, nT_fb] = size(feedback_hfa);
        [~, ~, nT_cue] = size(cue_hfa);
        acc_matrix = zeros(nT_fb, nT_cue);

        parfor t_fb = 1:nT_fb
            Xtrain = squeeze(feedback_hfa(:,:,t_fb));
            Xtrain = zscore(Xtrain, 0, 1);

            for t_cue = 1:nT_cue
                Xtest = squeeze(cue_hfa(:,:,t_cue));
                Xtest = zscore(Xtest, 0, 1);

                acc_matrix(t_fb,t_cue) = mv_classify(cfg, Xtrain, y_train, Xtest, y_train);
            end
        end

        all_acc(:,:,s) = acc_matrix;
        subj_ids{s} = subj;  % Track subject ID
    end

    %% --- REMOVE EMPTY SUBJECTS ---
    keep = squeeze(any(any(all_acc,1),2));
    all_acc = all_acc(:,:,keep);
    subj_ids = subj_ids(keep);  % Keep only subjects with valid data
    n_elec_subj = n_elec_subj(keep);  % Keep electrode counts for valid subjects
    fprintf('Kept %d subjects\n', sum(keep));

    %% --- GROUP-LEVEL STATS (vs. Chance = 1/3) ---
    chance = 1/3;
    nSubj_valid = size(all_acc, 3);
    
    % Vectorized t-test computation (much faster than nested loops)
    group_mean = mean(all_acc, 3);
    group_std = std(all_acc, 0, 3);
    tvals = (group_mean - chance) ./ (group_std / sqrt(nSubj_valid));
    tvals(~isfinite(tvals)) = 0;
    
    % Compute p-values (one-tailed, right)
    pvals = 1 - tcdf(tvals, nSubj_valid - 1);

    %% --- CLUSTER-BASED PERMUTATION TEST ---
    % Threshold at t corresponding to p<0.05 (one-tailed)
%     t_thresh = tinv(1-alpha, sum(keep)-1);
    t_thresh = 1;
    sig_mask = tvals > t_thresh;

    % Label clusters
    cc = bwconncomp(sig_mask, 8);
    cluster_tsum = cellfun(@(x) sum(tvals(x)), cc.PixelIdxList);

    % Null distribution via random sign-flip permutations (parallelized)
    % Pre-generate all random flips for reproducibility with parfor
    all_flips = (randi([0 1], [nPerm, nSubj_valid]) * 2 - 1);
    centered_acc = all_acc - chance;
    max_tsum = zeros(nPerm, 1);
    
    parfor p = 1:nPerm
        flips = all_flips(p, :);
        perm_data = centered_acc .* reshape(flips, 1, 1, []);
        perm_mean = mean(perm_data, 3);
        perm_std = std(perm_data, 0, 3);
        perm_t = perm_mean ./ (perm_std / sqrt(nSubj_valid));
        perm_t(~isfinite(perm_t)) = 0;
        perm_mask = perm_t > t_thresh;
        cc_perm = bwconncomp(perm_mask, 8);
        if cc_perm.NumObjects > 0
            max_tsum(p) = max(cellfun(@(x) sum(perm_t(x)), cc_perm.PixelIdxList));
        else
            max_tsum(p) = 0;
        end
    end

    % Compute cluster p-values
    cluster_pvals = arrayfun(@(x) mean(max_tsum >= x), cluster_tsum);
    sig_clusters = find(cluster_pvals < alpha);
    sig_mask_final = false(size(sig_mask));
    for c = sig_clusters
        sig_mask_final(cc.PixelIdxList{c}) = true;
    end

    %% --- STORE RESULTS FOR COMBINED PLOTTING ---
    all_acc_rois{r} = all_acc;
    sig_mask_rois{r} = sig_mask_final;
    time_fb_rois{r} = time_fb;
    time_cue_rois{r} = time_cue;
    valid_rois(r) = true;

    %% --- INDIVIDUAL ROI PLOT (optional, with updated aesthetics) ---
    fig_individual = figure('Position', [200 200 500 450]);
    set(fig_individual, 'Color', 'w');
    set(fig_individual, 'Renderer', 'painters');  % Better for PDF export
    
    imagesc(time_cue, time_fb, mean(all_acc, 3) * 100, plot_cfg.clim);
    axis xy;
    colormap(plot_cfg.colormap);
    
    % Colorbar
    cb = colorbar;
    cb.Label.String = 'classification accuracy (%)';
    cb.Label.FontName = plot_cfg.font_name;
    cb.Label.FontSize = plot_cfg.font_size_label;
    cb.FontName = plot_cfg.font_name;
    cb.FontSize = plot_cfg.font_size_tick;
    cb.TickDirection = plot_cfg.tick_dir;
    cb.LineWidth = plot_cfg.line_width;
    
    % Axis labels
    xlabel('time since recognition cue (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    ylabel('time since feedback (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
    
    % Tick configuration
    set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
    set(gca, 'TickDir', plot_cfg.tick_dir);
    set(gca, 'LineWidth', plot_cfg.line_width);
    set(gca, 'XTick', plot_cfg.xticks);
    set(gca, 'YTick', plot_cfg.yticks);
    set(gca, 'TickLength', plot_cfg.tick_length);
    box off;
    
    % ROI label in bottom-left corner
    text(0.05, 0.08, roi, 'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
        'FontSize', plot_cfg.font_size_roi + 2, 'FontWeight', 'bold', 'Color', plot_cfg.contour_color);
    
    % Significant cluster contours
    hold on;
    if any(sig_mask_final(:))
        contour(time_cue, time_fb, double(sig_mask_final), [0.5 0.5], plot_cfg.contour_color, 'LineWidth', plot_cfg.contour_width);
    end
    hold off;
    
    print(fig_individual, fullfile(output_dir, sprintf('%s_crossdecoding.pdf', roi)), '-dpdf', '-r300');

    save(fullfile(output_dir, sprintf('%s_all_acc_subjects.mat', roi)), 'all_acc', 'time_fb', 'time_cue', 'subj_ids', 'n_elec_subj', 'sig_mask_final', 'cluster_pvals', '-v7.3');
    
    %% --- SAVE DETAILED STATISTICAL REPORT ---
    txt_filename = fullfile(output_dir, sprintf('%s_crossdecoding_stats.txt', roi));
    
    fid = fopen(txt_filename, 'w');
    
    fprintf(fid, 'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
    fprintf(fid, '======================================================\n\n');
    
    %% --- GENERAL INFORMATION ---
    fprintf(fid, 'ROI: %s\n', roi);
    fprintf(fid, 'Dependent variable: RPE category\n');
    fprintf(fid, 'Number of subjects: %d\n', sum(keep));
    fprintf(fid, 'Smoothing window: %d ms\n', smooth_win_ms);
    fprintf(fid, 'Downsample factor: %d\n', downsample_factor);
    fprintf(fid, 'Time resolution (feedback): %.6f s\n', mean(diff(time_fb)));
    fprintf(fid, 'Time resolution (cue): %.6f s\n\n', mean(diff(time_cue)));
    
    %% --- DECODING CONFIGURATION ---
    fprintf(fid, 'Classifier: %s\n', cfg.classifier);
    fprintf(fid, 'Metric: %s\n', cfg.metric);
    fprintf(fid, 'Prior: %s\n', cfg.prior);
    fprintf(fid, 'Regularization: %s\n', cfg.reg);
    fprintf(fid, 'Cross-validation: %s (train on feedback, test on cue)\n', cfg.cv);
    fprintf(fid, 'HFA frequency range: 70-200 Hz\n');
    fprintf(fid, 'Z-scoring: per timepoint across trials (independent normalization)\n\n');

    %% --- STATISTICAL TEST DETAILS ---
    fprintf(fid, 'Statistical test: sign-flip permutation\n');
    fprintf(fid, 'Test statistic: t-test\n');
    fprintf(fid, 'Multiple comparison correction: cluster\n');
    fprintf(fid, 'Alpha level: %.4f\n', alpha);
    fprintf(fid, 'Number of permutations: %d\n\n', nPerm);
    
    fprintf(fid, 'Group-level design: within\n');
    fprintf(fid, 'Statistic: t-test\n');
    fprintf(fid, 'Tail: 1\n');
    fprintf(fid, 'Null hypothesis value: %.4f\n', chance);
    fprintf(fid, 'Permutation test: sign-flip\n');
    fprintf(fid, 'Cluster statistic: maxsum\n');
    fprintf(fid, 'Cluster-forming threshold: tinv(1-alpha, n-1) = %.4f\n', t_thresh);
    fprintf(fid, 'Cluster connectivity: 8-connected (2D)\n\n');
    
    %% --- CLUSTER RESULTS ---
    n_sig_clusters = numel(sig_clusters);
    fprintf(fid, 'Number of significant clusters: %d\n\n', n_sig_clusters);
    
    if n_sig_clusters == 0
        fprintf(fid, 'No significant clusters detected.\n');
    else
        for c = 1:numel(sig_clusters)
            cid = sig_clusters(c);
            [rows, cols] = ind2sub(size(sig_mask), cc.PixelIdxList{cid});
            
            t_fb_start = time_fb(min(rows));
            t_fb_end = time_fb(max(rows));
            t_cue_start = time_cue(min(cols));
            t_cue_end = time_cue(max(cols));
            
            fprintf(fid, 'Cluster %d\n', c);
            fprintf(fid, '  Feedback time range: %.4f – %.4f s\n', t_fb_start, t_fb_end);
            fprintf(fid, '  Cue time range: %.4f – %.4f s\n', t_cue_start, t_cue_end);
            fprintf(fid, '  Cluster-level p-value: %.6f\n', cluster_pvals(cid));
            fprintf(fid, '\n');
        end
    end
    
    fclose(fid);
    fprintf('Saved statistical summary:\n%s\n', txt_filename);
    
    %% --- SUBJECT-LEVEL REINSTATEMENT SCORES ---
    if any(sig_mask_final(:))
        nSubj = size(all_acc, 3);
        subj_reinstatement = zeros(nSubj, 1);
        subj_reinstatement_effect = zeros(nSubj, 1);  % accuracy - chance
        
        for subj_idx = 1:nSubj
            subj_acc = all_acc(:,:,subj_idx);
            % Mean accuracy within significant cluster
            subj_reinstatement(subj_idx) = mean(subj_acc(sig_mask_final));
            % Effect size: accuracy above chance
            subj_reinstatement_effect(subj_idx) = mean(subj_acc(sig_mask_final)) - chance;
        end
        
        % Display results to console
        fprintf('\nSubject-level reinstatement scores for %s:\n', roi);
        fprintf('%-10s\t%s\t\t%s\t\t%s\n', 'Subject', 'Mean Acc', 'Effect', 'nElec');
        for subj_idx = 1:nSubj
            fprintf('%-10s\t%.4f\t\t%.4f\t\t%d\n', subj_ids{subj_idx}, subj_reinstatement(subj_idx), subj_reinstatement_effect(subj_idx), n_elec_subj(subj_idx));
        end
        
        % Check correlation between reinstatement and electrode count
        [r_elec, p_elec] = corr(subj_reinstatement_effect, n_elec_subj(:), 'Type', 'Spearman');
        fprintf('\nCorrelation between reinstatement effect and electrode count:\n');
        fprintf('  Spearman rho = %.4f, p = %.4f\n', r_elec, p_elec);
        if p_elec < 0.05
            warning('Significant correlation between reinstatement and electrode count - consider controlling for this in downstream analyses.');
        end
        
        % Save as CSV with subject IDs and electrode counts
        subj_table = table(subj_ids(:), subj_reinstatement, subj_reinstatement_effect, n_elec_subj(:), ...
            'VariableNames', {'Subject', 'MeanAccuracy', 'ReinstateEffect', 'NumElectrodes'});
        writetable(subj_table, fullfile(output_dir, sprintf('%s_subject_reinstatement.csv', roi)));
        fprintf('Saved subject reinstatement scores:\n%s\n', ...
            fullfile(output_dir, sprintf('%s_subject_reinstatement.csv', roi)));
        
        % Bar plot of subject contributions
        fig_bar = figure('Position', [100 100 700 400]);
        set(fig_bar, 'Color', 'w');
        set(fig_bar, 'Renderer', 'painters');
        bar(subj_reinstatement_effect, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'none');
        hold on;
        yline(0, 'k--', 'LineWidth', plot_cfg.line_width);
        xlabel('Subject', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
        ylabel('Reinstatement Effect (Accuracy - Chance)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
        title(sprintf('%s: Subject-Level Reinstatement', roi), 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label + 2);
        set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        set(gca, 'TickDir', plot_cfg.tick_dir);
        set(gca, 'LineWidth', plot_cfg.line_width);
        set(gca, 'TickLength', plot_cfg.tick_length);
        xticks(1:nSubj);
        xticklabels(subj_ids);
        xtickangle(45);
        xlim([0.5 nSubj+0.5]);
        box off;
        hold off;
        print(fig_bar, fullfile(output_dir, sprintf('%s_subject_reinstatement.pdf', roi)), '-dpdf', '-r300');
    else
        fprintf('\nNo significant clusters found for %s - skipping subject-level reinstatement.\n', roi);
    end
end

%% ============================================================
%  COMBINED SUBPLOT FIGURE (Publication-quality)
% =============================================================

% Identify which ROIs have valid data
valid_idx = find(valid_rois);
nValidROIs = numel(valid_idx);

if nValidROIs > 0
    fprintf('\n===== Creating combined subplot figure =====\n');
    
    % Figure sizing: adjust width based on number of ROIs
    subplot_width = 220;  % pixels per subplot
    fig_width = subplot_width * nValidROIs + 120;  % extra space for colorbar
    fig_height = 350;
    
    fig_combined = figure('Position', [100 100 fig_width fig_height]);
    set(fig_combined, 'Color', 'w');
    set(fig_combined, 'Renderer', 'painters');  % Better for PDF export
    
    % Create tiled layout for better control
    t = tiledlayout(1, nValidROIs, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    for i = 1:nValidROIs
        r = valid_idx(i);
        roi_name = ROIs{r};
        
        nexttile;
        
        % Get data for this ROI
        acc_data = mean(all_acc_rois{r}, 3) * 100;  % Convert to percentage
        sig_mask = sig_mask_rois{r};
        t_cue = time_cue_rois{r};
        t_fb = time_fb_rois{r};
        
        % Plot heatmap
        imagesc(t_cue, t_fb, acc_data, plot_cfg.clim);
        axis xy;
        colormap(plot_cfg.colormap);
        
        % Axis configuration
        set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
        set(gca, 'TickDir', plot_cfg.tick_dir);
        set(gca, 'LineWidth', plot_cfg.line_width);
        set(gca, 'XTick', plot_cfg.xticks);
        set(gca, 'YTick', plot_cfg.yticks);
        set(gca, 'TickLength', plot_cfg.tick_length);
        box off;
        
        % X-axis label for all subplots
        xlabel('time since recognition cue (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label - 1);
        
        % Y-axis label only for first subplot
        if i == 1
            ylabel('time since feedback (s)', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label - 1);
        else
            set(gca, 'YTickLabel', []);
        end
        
        % ROI label in bottom-left corner
        text(0.06, 0.06, roi_name, 'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
            'FontSize', plot_cfg.font_size_roi, 'FontWeight', 'bold', 'Color', plot_cfg.contour_color, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
        
        % Significant cluster contours
        hold on;
        if any(sig_mask(:))
            contour(t_cue, t_fb, double(sig_mask), [0.5 0.5], plot_cfg.contour_color, 'LineWidth', plot_cfg.contour_width);
        end
        hold off;
    end
    
    % Shared colorbar
    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Label.String = 'classification accuracy (%)';
    cb.Label.FontName = plot_cfg.font_name;
    cb.Label.FontSize = plot_cfg.font_size_label - 1;
    cb.FontName = plot_cfg.font_name;
    cb.FontSize = plot_cfg.font_size_tick;
    cb.TickDirection = plot_cfg.tick_dir;
    cb.LineWidth = plot_cfg.line_width;
    
    % Save combined figure
    print(fig_combined, fullfile(output_dir, 'all_ROIs_crossdecoding_combined.pdf'), '-dpdf', '-r300');
    saveas(fig_combined, fullfile(output_dir, 'all_ROIs_crossdecoding_combined.fig'));
    fprintf('Saved combined figure:\n%s\n', fullfile(output_dir, 'all_ROIs_crossdecoding_combined.pdf'));
end
