%% --- CONFIGURATION ---
base_dir = '/Volumes/T7 Shield';
load_dir = fullfile(base_dir, 'work/qasims01/MemoryBanditData/EMU');
output_dir = fullfile(base_dir, 'scratch/MemoryBandit/MVPA/decoding_feedback/');

% ROIs = {['dlPFC', 'OFC', 'ACC', 'dmPFC']};  % list of ROIs
% 'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY', 'Temporal','PFC'
ROIs = {'OFC', 'ACC', 'dmPFC', 'dlPFC', 'HPC', 'AMY', 'Temporal','PFC'};
subjects = {'MS012', 'UI001', 'MS016', 'MS017', 'UI002', 'MS019', 'MS020', 'MS022', ...
            'MS023', 'UI003', 'MS025', 'MS026', 'MS028', 'MS030', 'UI004', ...
            'MS035', 'MS036', 'UI006', 'UI007'};

downsample_factor = 5;            % match your Python version
win_ms = 150;                     % smoothing window in ms
permutation = 0;
nPerm = 250;                         % number of permutations for significance
alpha = 0.05;                     % cluster-level significance
% thresh = 0.5;
% min_cluster_pts = round(0.05 * 100);  % e.g., fs=250 => 38 points

%% --- LOAD TABLES ONCE ---
learn_df = readtable(fullfile(load_dir, 'learn_df_RWH.csv'));
combined_df = readtable(fullfile(load_dir, 'full_df_RWH.csv'));

%% --- LOOP OVER DEPENDENT VARIABLES ---
% dependent_vars = {'reward', 'phit'};  % 'reward' or 'rpe_category'
dependent_vars = {'rpe_category'}; 
for d = 1:numel(dependent_vars)
    dependent_var = dependent_vars{d};
    fprintf('Processing variable: %s\n', dependent_var);
    %% --- LOOP OVER ROIs ---
    for r = 1:numel(ROIs)
%         roi = ROIs{r};
        if strcmp(ROIs{r}, 'PFC')
            roi = {'OFC', 'ACC', 'dlPFC', 'dmPFC'};
        elseif strcmp(ROIs{r}, 'MTL')
            roi = {'HPC', 'AMY', 'PHC', 'EC'};
        else
            roi = ROIs{r};
        end

        fprintf('Processing ROI: %s\n', ROIs{r} );
        
        % Preallocate arrays for all subjects
        max_timepoints = 1000; % placeholder; will be adjusted dynamically
        all_acc_subjects = [];
        all_perm_subjects = [];
        all_results_subjects = cell(1, numel(subjects));  % store results struct
        
        %% --- LOOP OVER SUBJECTS ---
        for s = 1:numel(subjects)
            subj = subjects{s};
            fprintf('  Subject: %s\n', subj);
            
            %% --- SELECT SUBJECT DATA ---
            subj_learn_df = learn_df(strcmp(learn_df.participant, subj), :);
            subj_combined_df = combined_df(strcmp(combined_df.condition,'Day1') & strcmp(combined_df.participant, subj), :);
            subj_combined_df = subj_combined_df(~isnan(subj_combined_df.trials_gamble), :);
    
            %% Add memory information
            subj_combined_df.phit = double(strcmp(subj_combined_df.hits, 'True'));
    
            %% Process RPE categories using tercile split within subject
            edges_learn = quantile(subj_learn_df.rpe, [0 1/3 2/3 1]);
            subj_learn_df.rpe_category = discretize(subj_learn_df.rpe, edges_learn, 'categorical', {'negative', 'neutral', 'positive'});
            edges_combined = quantile(subj_combined_df.rpe, [0 1/3 2/3 1]);
            subj_combined_df.rpe_category = discretize(subj_combined_df.rpe, edges_combined, 'categorical', {'negative', 'neutral', 'positive'});
    
            %% Trial mapping and remove fast RTs
            if strcmp(dependent_var, 'phit')
                encoding_trials = subj_combined_df.trials_gamble;
%                 mask_rt = subj_combined_df.gamble_rt_gamble >= 0.3;
%                 encoding_trials = encoding_trials(mask_rt);
            else
                encoding_trials = subj_learn_df.trials;
%                 mask_rt = subj_learn_df.gamble_rt >= 0.3;
%                 encoding_trials = encoding_trials(mask_rt);
            end

%             %% Trial mapping
%             if strcmp(dependent_var, 'phit')
%                 encoding_trials = subj_combined_df.trials_gamble;
%             else
%                 encoding_trials = subj_learn_df.trials;
%             end
    
            if isempty(encoding_trials)
                warning('No trials for subject %s', subj);
                continue;
            end
    
            %% Load electrode info
            elec_df = readtable(fullfile(base_dir, 'projects','guLab','Salman','EphysAnalyses', subj, 'Day1_reref_elec_df'));
%             roi_electrodes = elec_df.label(strcmp(elec_df.salman_region, roi));
            roi_electrodes = elec_df.label(find(matches(elec_df.salman_region,roi)));
            if isempty(roi_electrodes)
                warning('No electrodes found for %s in %s', subj, ROIs{r} );
                continue;
            end
    
            %% Load feedback TFR data
            filepath = fullfile(base_dir,'projects','guLab','Salman','EphysAnalyses',subj,'scratch','TFR');
            fb = load(fullfile(filepath, 'feedback_start-tfr.mat'));
            feedback_tfr.powspctrm = fb.powspctrm;
            feedback_tfr.freq = fb.freqs;
            feedback_tfr.time = fb.times;
            feedback_tfr.label = fb.ch_names;
            feedback_tfr.fsample = fb.sfreq;
    
            %% Select ROI electrodes
            [~, idx_fb] = ismember(roi_electrodes, feedback_tfr.label);
            feedback_tfr.powspctrm = feedback_tfr.powspctrm(:, idx_fb, :, :);
    
            %% Extract HFA 70–200 Hz
            hfa_mask = feedback_tfr.freq >= 70 & feedback_tfr.freq <= 201;
            feedback_hfa = squeeze(mean(feedback_tfr.powspctrm(:, :, hfa_mask, :), 3));  % trials x electrodes x time
            Ndims = ndims(feedback_hfa); 
            if Ndims < 3 
                feedback_hfa = reshape(feedback_hfa, [size(feedback_hfa, 1), 1, size(feedback_hfa, 2)]); 
            end
            %% --- SMOOTH (VECTORIZE) ---
            fs = feedback_tfr.fsample;
            win_size = round(fs * (win_ms/1000));
            feedback_hfa_smooth = movmean(feedback_hfa, win_size, 3);  % vectorized smoothing

            %% --- CLIP AND INTERPOLATE ARTIFACT VALUES ---
            z_thresh = 10;
            [nTrials_raw, nElec, nTime] = size(feedback_hfa_smooth);
            
            for trial = 1:nTrials_raw
                for elec = 1:nElec
                    trace = squeeze(feedback_hfa_smooth(trial, elec, :));
                    bad_idx = abs(trace) > z_thresh;
                    
                    if any(bad_idx)
                        good_idx = find(~bad_idx);
                        bad_idx_list = find(bad_idx);
                        
                        if numel(good_idx) >= 2
                            % Interpolate bad points from good neighbors
                            trace(bad_idx) = interp1(good_idx, trace(good_idx), bad_idx_list, 'linear', 'extrap');
                        else
                            % Fallback: clip if not enough good points to interpolate
                            trace(trace > z_thresh) = z_thresh;
                            trace(trace < -z_thresh) = -z_thresh;
                        end
                        
                        feedback_hfa_smooth(trial, elec, :) = trace;
                    end
                end
            end
    
            %% Select encoding trials
            feedback_hfa_smooth = feedback_hfa_smooth(encoding_trials,:,:);
    
            %% Downsample
            feedback_hfa_ds = feedback_hfa_smooth(:,:,1:downsample_factor:end);
    
        %% Define labels
        switch dependent_var
            case 'reward'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.5;
            case 'rpe'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0;
            case 'phit'
                y = subj_combined_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.5;
            case 'rpe_category'
                y = subj_learn_df.(dependent_var);
%                 y = y(mask_rt);
                chancelevel = 0.33;
        end
            y = double(y);
    
            %% --- MVPA-LIGHT CONFIGURATION ---
            cfg = [];
            switch dependent_var
                case 'reward'
                    cfg.classifier = 'lda'; cfg.metric = 'acc';
                    cfg.cv = 'leaveout'; 
                case 'phit'
                    cfg.classifier = 'lda'; cfg.metric = 'acc';
                    cfg.cv = 'leaveout'; 
                case 'rpe_category'
                    cfg.classifier = 'multiclass_lda'; 
                    cfg.metric = 'acc'; 
                    cfg.prior = 'uniform';
                    cfg.reg = 'ridge';
                    cfg.cv = 'leaveout'; 
                case 'rpe'
                    cfg.reg = 'ridge'; 
                    cfg.metric = 'r_squared';
                    cfg.hyperparameter = [];
                    cfg.hyperparameter = 'lambda';
                    cfg.lambda = 'auto';
                    cfg.dimension_names = {'samples' 'channels', 'time points'};
            end
            cfg.feedback = 0;
    
            %% --- TIME-RESOLVED DECODING ---
            if strcmp(dependent_var, 'rpe')
                [perf, results] = mv_regress(cfg, feedback_hfa_ds, y);
            else
                [perf, results] = mv_classify_across_time(cfg, feedback_hfa_ds, y);
            end
    
            % Preallocate arrays if first subject
            if isempty(all_acc_subjects)
                num_timepoints = size(perf,1);
                all_acc_subjects = zeros(num_timepoints, numel(subjects));
                all_perm_subjects = zeros(num_timepoints, nPerm, numel(subjects));
            end
            all_acc_subjects(:,s) = perf;
            all_results_subjects{s} = results;
    
            %% --- PERMUTATIONS (PARALLEL) ---
            if permutation == 1
                perm_metrics = zeros(size(perf,1), nPerm);
                parfor p = 1:nPerm
                    y_perm = y(randperm(length(y)));
                    if strcmp(dependent_var, 'rpe')
                        [perf_perm, ~] = mv_regress(cfg, feedback_hfa_ds, y_perm);
                    else
                        [perf_perm, ~] = mv_classify_across_time(cfg, feedback_hfa_ds, y_perm);
                    end
                    perm_metrics(:,p) = perf_perm;
                end
                all_perm_subjects(:,:,s) = perm_metrics;
            end
          
        end
    
        %% --- REMOVE SUBJECTS WITH ALL-ZERO ACCURACY ---
        subjects_to_keep = any(all_acc_subjects ~= 0, 1);
        all_acc_subjects = all_acc_subjects(:,subjects_to_keep);
        all_results_subjects = all_results_subjects(subjects_to_keep);
    
        if permutation == 1
            all_perm_subjects = all_perm_subjects(:, :, subjects_to_keep);
    
            %% --- ZSCORE AGAINST PERMUTATIONS (VECTORIZE) ---
            mu_perm = mean(all_perm_subjects,2);
            sigma_perm = std(all_perm_subjects,0,2);
            zscore_acc = (all_acc_subjects - squeeze(mu_perm)) ./ squeeze(sigma_perm);
    
            for n = 1:length(all_results_subjects)
                all_results_subjects{1, n}.perf = zscore_acc(:, n);
            end
        
            % Average across subjects
            mean_z = mean(zscore_acc,2);
            sem_z  = std(zscore_acc,0,2)/sqrt(size(zscore_acc,2));
    %     
        end
%         time_vec = feedback_tfr.time(1:downsample_factor:end);
%         cfg_stat = [];
%         cfg_stat.metric          =  cfg.metric;
%         cfg_stat.test            = 'permutation';
%         cfg_stat.correctm        = 'cluster';  % correction method is cluster
%         cfg_stat.n_permutations  = 1000;
%         % Clusterstatistic is the actual statistic used for the clustertest.
%         % Normally the default value 'maxum' is used, we are setting it here
%         % explicitly for clarity. Maxsum adds up the statistics calculated at each
%         % time point (the latter are set below using cfg_stat.statistic)
%         cfg_stat.clusterstatistic = 'maxsum';
%         cfg_stat.alpha           = 0.05; % use standard significance threshold of 5%
%         % Level 2 stats design: we have to choose between within-subject and
%         % between-subjects. Between-subjects is relevant when there is two
%         % different experimental groups (eg patients vs controls) and we want to
%         % investigate whether their MVPA results are significantly different. Here,
%         % we have only one group and we want to see whether the AUC is
%         % significantly different from a null value, hence the statistical design
%         % is within-subject
%         cfg_stat.design          = 'within';
%         % cfg_stat.statistic defines how the difference between the AUC values and
%         % the null is calculated at each time point (across subjects).
%         % We can choose t-test or its nonparametric counterpart Wilcoxon test. We
%         % choose Wilcoxon here.
%         cfg_stat.statistic       = 'wilcoxon';
%         % The null value for AUC (corresponding to a random classifier) is 0.5
%         if permutation == 1
%             cfg_stat.null = 0;   % z-scored null
%         else
%             cfg_stat.null = chancelevel;
%         end
%         % clustercritval is a parameter that needs to be selected by the user. In a
%         % Level 2 (group level) test, it represents the critical cutoff value for
%         % the statistic. Here, we selected Wilcoxon, so clustercritval corresponds
%         % to the cutoff value for the z-statistic which is obtained by a normal
%         % approximation.
%         cfg_stat.clustercritval= 1.645;
%         cfg_stat.tail = 1;
%         % z-val = 1.65 corresponds to uncorrected p-value = 0.1
%         % z-val = 1.96 corresponds to uncorrected p-value = 0.05
%         % z-val = 2.58 corresponds to uncorrected p-value = 0.01
%         % Note that these z-values correspond to a two-sided test. In many cases it
%         % suffices to use a one-sided test. In this case, the following cutoff
%         % values are obtained:
%         % z-val = 1.282 corresponds to uncorrected p-value = 0.1
%         % z-val = 1.645 corresponds to uncorrected p-value = 0.05
%         % z-val = 2.326 corresponds to uncorrected p-value = 0.01
%         stat_level2 = mv_statistics(cfg_stat, all_results_subjects);
%         result_average = mv_combine_results(all_results_subjects, 'average');
%         result_average.perf_std{1, 1} = result_average.perf_std{1, 1} / sqrt(length(all_results_subjects));  % turn this into standard error for plots
%         mv_plot_result(result_average, time_vec, 'mask', stat_level2.mask)
%     %         grid on;
%     %         set(gcf,'units','centimeters','position',[xini,yini,xsize,ysize]); 
%         set(gca,'FontSize',10,'FontName','Arial');
%     %         print("title",'-depsc2');
%     %         saveas(gcf, fullfile(output_dir,[roi '_phit.eps']));
%         f = gcf;
%         exportgraphics(f,fullfile(output_dir,[ROIs{r} '_' dependent_var '_' num2str(win_ms) '.pdf']),'Resolution',300)
                time_vec = feedback_tfr.time(1:downsample_factor:end);
        
        %% --- GROUP-LEVEL STATS (vs. Chance) ---
        % all_acc_subjects is [timepoints x subjects]
        nSubj = size(all_acc_subjects, 2);
        nTime = size(all_acc_subjects, 1);
        
        tvals = zeros(nTime, 1);
        pvals = zeros(nTime, 1);
        
        for t = 1:nTime
            [~, p, ~, stats] = ttest(all_acc_subjects(t,:), chancelevel, 'Tail', 'right');
            tvals(t) = stats.tstat;
            pvals(t) = p;
        end
        
        %% --- CLUSTER-BASED PERMUTATION TEST ---
        % Threshold at t corresponding to p<0.05 (one-tailed)
%         t_thresh = tinv(1 - alpha, nSubj - 1);
        t_thresh = 1;
        sig_mask = tvals > t_thresh;
        
        % Label clusters (1D connectivity)
        cc = bwconncomp(sig_mask, 4);
        cluster_tsum = cellfun(@(x) sum(tvals(x)), cc.PixelIdxList);
        
        % Null distribution via random sign-flip permutations
        nPerm_stat = 1000;
        max_tsum = zeros(nPerm_stat, 1);
        for p = 1:nPerm_stat
            flips = (randi([0 1], [1, nSubj]) * 2 - 1);
            perm_data = all_acc_subjects - chancelevel;
            perm_data = perm_data .* flips;
            perm_t = mean(perm_data, 2) ./ (std(perm_data, 0, 2) / sqrt(nSubj));
            perm_mask = perm_t > t_thresh;
            cc_perm = bwconncomp(perm_mask, 4);
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
        n_sig_clusters = numel(sig_clusters);
        
        %% --- PLOT RESULTS ---
        figure('Position', [200 200 800 400]);
        hold on;
        
        % Plot mean accuracy with SEM
        mean_acc = mean(all_acc_subjects, 2);
        sem_acc = std(all_acc_subjects, 0, 2) / sqrt(nSubj);
        
        fill([time_vec, fliplr(time_vec)], ...
             [mean_acc + sem_acc; flipud(mean_acc - sem_acc)]', ...
             [0.7 0.7 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
        plot(time_vec, mean_acc, 'b', 'LineWidth', 2);
        yline(chancelevel, 'k--', 'LineWidth', 1);
        
        % Highlight significant clusters
        if any(sig_mask_final)
            sig_times = time_vec(sig_mask_final);
            yl = ylim;
            for i = 1:length(sig_times)
                plot([sig_times(i) sig_times(i)], yl, 'r', 'LineWidth', 2);
            end
        end
        
        xlabel('Time (s)');
        ylabel('Accuracy');
        title(sprintf('%s: %s decoding', ROIs{r}, dependent_var));
        set(gca, 'FontSize', 10, 'FontName', 'Arial');
        ylim([0.26 0.42]); % Sets y-axis from -2 to 2
        hold off;
        
        f = gcf;
        exportgraphics(f, fullfile(output_dir, [ROIs{r} '_' dependent_var '_' num2str(win_ms) '.pdf']), 'Resolution', 300)

        %% --- SAVE DETAILED STATISTICAL REPORT ---
        txt_filename = fullfile(output_dir, ...
            [ROIs{r} '_' dependent_var '_' num2str(win_ms) '_stats.txt']);
        
        fid = fopen(txt_filename,'w');
        
        fprintf(fid,'STATISTICAL REPORT: MVPA CLUSTER-BASED PERMUTATION TEST\n');
        fprintf(fid,'======================================================\n\n');
        
        %% --- GENERAL INFORMATION ---
        fprintf(fid,'ROI: %s\n', ROIs{r});
        fprintf(fid,'Dependent variable: %s\n', dependent_var);
        fprintf(fid,'Number of subjects: %d\n', nSubj);
        fprintf(fid,'Smoothing window: %d ms\n', win_ms);
        fprintf(fid,'Downsample factor: %d\n', downsample_factor);
        fprintf(fid,'Time resolution: %.6f s\n\n', mean(diff(time_vec)));
        
        %% --- DECODING CONFIGURATION ---
        fprintf(fid,'Classifier: %s\n', cfg.classifier);
        fprintf(fid,'Metric: %s\n', cfg.metric);
        fprintf(fid,'Cross-validation: %s\n', cfg.cv);
        fprintf(fid,'HFA frequency range: 70-200 Hz\n\n');
        
        %% --- STATISTICAL TEST DETAILS ---
        fprintf(fid,'Statistical test: sign-flip permutation\n');
        fprintf(fid,'Test statistic: t-test\n');
        fprintf(fid,'Multiple comparison correction: cluster\n');
        fprintf(fid,'Alpha level: %.4f\n', alpha);
        fprintf(fid,'Number of permutations: %d\n\n', nPerm_stat);
        
        fprintf(fid,'Group-level design: within\n');
        fprintf(fid,'Statistic: t-test\n');
        fprintf(fid,'Tail: 1\n');
        fprintf(fid,'Null hypothesis value: %.4f\n', chancelevel);
        fprintf(fid,'Permutation test: sign-flip\n');
        fprintf(fid,'Cluster statistic: maxsum\n');
        fprintf(fid,'Cluster-forming threshold: %.4f\n', t_thresh);
        fprintf(fid,'Cluster connectivity: 4-connected (1D)\n\n');
        
        %% --- CLUSTER RESULTS ---
        fprintf(fid,'Number of significant clusters: %d\n\n', n_sig_clusters);
        
        if n_sig_clusters == 0
            fprintf(fid,'No significant clusters detected.\n');
        else
            for c = 1:numel(sig_clusters)
                cid = sig_clusters(c);
                cluster_idx = cc.PixelIdxList{cid};
                
                t_start = time_vec(min(cluster_idx));
                t_end   = time_vec(max(cluster_idx));
        
                fprintf(fid,'Cluster %d\n', c);
                fprintf(fid,'  Time range: %.4f – %.4f s\n', t_start, t_end);
                fprintf(fid,'  Cluster-level p-value: %.6f\n', cluster_pvals(cid));
                fprintf(fid,'\n');
            end
        end
        
        fclose(fid);
        
        fprintf('Saved statistical summary:\n%s\n', txt_filename);

        %% --- SAVE DECODING RESULTS ---
        mat_filename = fullfile(output_dir, [ROIs{r} '_' dependent_var '_' num2str(win_ms) '_results.mat']);
        subjects_used = subjects(subjects_to_keep);
        save(mat_filename, 'all_acc_subjects', 'subjects_used', 'time_vec', ...
             'chancelevel', 'win_ms', 'downsample_factor', 'dependent_var', 'cfg');
        fprintf('Saved decoding results:\n%s\n', mat_filename);

    end
end