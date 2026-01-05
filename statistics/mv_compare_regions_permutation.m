function results = mv_compare_regions_permutation(output_dir, ROIs, varargin)
% MV_COMPARE_REGIONS_PERMUTATION - Compare cross-decoding across regions using sign-flip permutation
%
% Uses the same sign-flip cluster-based permutation approach as within-region
% tests, but applied to the DIFFERENCE between regions. Tests whether a
% reference region (e.g., ACC) shows significantly higher cross-decoding
% than other regions.
%
% Usage:
%   results = mv_compare_regions_permutation(output_dir, ROIs)
%   results = mv_compare_regions_permutation(output_dir, ROIs, 'Name', Value, ...)
%
% Inputs:
%   output_dir  - Directory containing saved '*_all_acc_subjects.mat' files
%   ROIs        - Cell array of ROI names (e.g., {'OFC','ACC','dmPFC',...})
%
% Optional Name-Value Parameters:
%   'reference_roi'  - ROI to compare against others (default: 'ACC')
%   'use_ref_cluster' - Restrict analysis to reference ROI's significant cluster (default: true)
%   'nPerm'          - Number of permutations (default: 1000)
%   't_thresh'       - Cluster-forming threshold (default: 1)
%   'alpha'          - Significance level (default: 0.05)
%   'seed'           - Random seed for reproducibility (default: 42)
%   'save_results'   - Save results to file (default: true)
%
% Note: This function requires that subjects are aligned across region files.
%       If your saved files don't contain subject identifiers, the function
%       assumes subjects are in the same order across files (matching by index).
%       For best results, save subject IDs when running the main analysis.
%
% Output:
%   results - Structure containing:
%       .pairwise            - Table with cluster stats, p-values per comparison
%       .diff_matrices       - Cell array of difference matrices per comparison
%       .observed_clusters   - Observed cluster info per comparison
%       .null_distributions  - Null max cluster distributions
%
% Author: Generated for cross-decoding analysis
% Date: 2026

%% Parse inputs
p = inputParser;
addRequired(p, 'output_dir', @ischar);
addRequired(p, 'ROIs', @iscell);
addParameter(p, 'reference_roi', 'ACC', @ischar);
addParameter(p, 'use_ref_cluster', true, @islogical);
addParameter(p, 'nPerm', 1000, @isnumeric);
addParameter(p, 't_thresh', 1, @isnumeric);
addParameter(p, 'alpha', 0.05, @isnumeric);
addParameter(p, 'seed', 42, @isnumeric);
addParameter(p, 'save_results', true, @islogical);
parse(p, output_dir, ROIs, varargin{:});

opts = p.Results;
nROIs = numel(ROIs);

% Set random seed for reproducibility
rng(opts.seed);
fprintf('Random seed set to: %d\n', opts.seed);

%% Find reference ROI index
ref_idx = find(strcmp(ROIs, opts.reference_roi));
if isempty(ref_idx)
    error('Reference ROI "%s" not found in ROIs list.', opts.reference_roi);
end

%% Load all region data
fprintf('Loading data from %d regions...\n', nROIs);

region_data = cell(nROIs, 1);
region_subj_ids = cell(nROIs, 1);
nSubj_per_region = zeros(nROIs, 1);
has_subj_ids = true;

for r = 1:nROIs
    fname = fullfile(output_dir, sprintf('%s_all_acc_subjects.mat', ROIs{r}));
    if ~exist(fname, 'file')
        error('File not found: %s', fname);
    end
    data = load(fname);
    region_data{r} = data.all_acc;
    [nT_fb, nT_cue, nSubj_per_region(r)] = size(data.all_acc);
    
    % Load subject IDs if available
    if isfield(data, 'subj_ids')
        region_subj_ids{r} = data.subj_ids;
    else
        has_subj_ids = false;
        region_subj_ids{r} = {};
    end
    
    fprintf('  %s: [%d × %d × %d subjects]\n', ROIs{r}, nT_fb, nT_cue, nSubj_per_region(r));
end

time_fb = data.time_fb;
time_cue = data.time_cue;

if ~has_subj_ids
    warning(['Subject IDs not found in saved files. Assuming subjects are aligned by index. ' ...
             'Re-run test_script_crossdecoding.m to save subject IDs for accurate matching.']);
end

%% Check subject alignment
ref_data = region_data{ref_idx};
ref_subj_ids = region_subj_ids{ref_idx};
nSubj_ref = nSubj_per_region(ref_idx);

fprintf('\nReference region (%s) has %d subjects.\n', opts.reference_roi, nSubj_ref);

%% Load reference ROI's significant cluster mask (if using)
ref_cluster_mask = [];
use_within_cluster = false;  % Track whether we're actually using within-cluster test

if opts.use_ref_cluster
    ref_fname = fullfile(output_dir, sprintf('%s_all_acc_subjects.mat', opts.reference_roi));
    ref_file = load(ref_fname);
    
    if isfield(ref_file, 'sig_mask_final')
        ref_cluster_mask = ref_file.sig_mask_final;
        
        % Check if it's a valid logical mask with significant pixels
        if islogical(ref_cluster_mask) && any(ref_cluster_mask(:))
            n_sig_pixels = sum(ref_cluster_mask(:));
            use_within_cluster = true;
            
            % Get time window info
            [rows, cols] = find(ref_cluster_mask);
            fprintf('\nUsing %s significant cluster (%d pixels):\n', opts.reference_roi, n_sig_pixels);
            fprintf('  Feedback time: %.3f – %.3f s\n', time_fb(min(rows)), time_fb(max(rows)));
            fprintf('  Cue time: %.3f – %.3f s\n', time_cue(min(cols)), time_cue(max(cols)));
        else
            warning('Reference ROI %s has no significant cluster. Using full time×time matrix.', opts.reference_roi);
            ref_cluster_mask = [];
        end
    else
        warning(['sig_mask_final not found in %s file. Re-run test_script_crossdecoding.m. ' ...
                 'Using full time×time matrix.'], opts.reference_roi);
    end
end

if use_within_cluster
    fprintf('\nTest mode: within-cluster\n');
else
    fprintf('\nTest mode: cluster-based (full matrix)\n');
end

%% Run pairwise comparisons
fprintf('\n=== Pairwise Permutation Tests (%s vs Others) ===\n', opts.reference_roi);

pairwise = table();
pairwise.ROI = cell(nROIs-1, 1);
pairwise.nSubj_paired = zeros(nROIs-1, 1);
pairwise.observed_cluster_stat = zeros(nROIs-1, 1);
pairwise.cluster_p = zeros(nROIs-1, 1);
pairwise.mean_diff = zeros(nROIs-1, 1);
pairwise.sig = false(nROIs-1, 1);

diff_matrices = cell(nROIs-1, 1);
observed_clusters = cell(nROIs-1, 1);
null_distributions = cell(nROIs-1, 1);

idx = 1;
for r = 1:nROIs
    if r == ref_idx
        continue;
    end
    
    other_data = region_data{r};
    other_subj_ids = region_subj_ids{r};
    nSubj_other = nSubj_per_region(r);
    
    % Find overlapping subjects between reference and other region
    if has_subj_ids && ~isempty(ref_subj_ids) && ~isempty(other_subj_ids)
        % Use subject IDs to find intersection
        [common_subjs, ref_idx_match, other_idx_match] = intersect(ref_subj_ids, other_subj_ids);
        nSubj_paired = numel(common_subjs);
        
        if nSubj_paired > 0
            % Extract matched data
            ref_matched = ref_data(:,:,ref_idx_match);
            other_matched = other_data(:,:,other_idx_match);
        end
    else
        % Fallback: assume aligned by index, use minimum
        nSubj_paired = min(nSubj_ref, nSubj_other);
        ref_matched = ref_data(:,:,1:nSubj_paired);
        other_matched = other_data(:,:,1:nSubj_paired);
        common_subjs = {};
    end
    
    if nSubj_paired < 3
        warning('Only %d paired subjects for %s vs %s. Skipping.', ...
            nSubj_paired, opts.reference_roi, ROIs{r});
        pairwise.ROI{idx} = ROIs{r};
        pairwise.nSubj_paired(idx) = nSubj_paired;
        pairwise.observed_cluster_stat(idx) = NaN;
        pairwise.cluster_p(idx) = NaN;
        pairwise.mean_diff(idx) = NaN;
        pairwise.sig(idx) = false;
        idx = idx + 1;
        continue;
    end
    
    fprintf('\n--- %s vs %s (%d paired subjects) ---\n', ...
        opts.reference_roi, ROIs{r}, nSubj_paired);
    
    % Compute difference matrix: reference - other
    % Shape: [nT_fb × nT_cue × nSubj_paired]
    diff_acc = ref_matched - other_matched;
    diff_matrices{idx} = diff_acc;
    
    %% Test depends on whether we're using reference cluster or full matrix
    if use_within_cluster
        %% WITHIN-CLUSTER TEST: Mean difference within reference's significant cluster
        % Extract values within cluster for each subject
        subj_diff_in_cluster = zeros(nSubj_paired, 1);
        for s = 1:nSubj_paired
            subj_diff_matrix = diff_acc(:,:,s);
            subj_diff_in_cluster(s) = mean(subj_diff_matrix(ref_cluster_mask));
        end
        
        % Observed t-statistic
        obs_mean_diff = mean(subj_diff_in_cluster);
        obs_std_diff = std(subj_diff_in_cluster);
        obs_t_stat = obs_mean_diff / (obs_std_diff / sqrt(nSubj_paired));
        
        fprintf('  Mean diff within cluster: %.4f (t = %.3f)\n', obs_mean_diff, obs_t_stat);
        
        % Store for output
        overall_mean_diff = obs_mean_diff;
        obs_max_cluster = obs_t_stat;  % Use t-stat as the test statistic
        cluster_info = struct('mean_diff', obs_mean_diff, 't_stat', obs_t_stat, ...
            'n_pixels', sum(ref_cluster_mask(:)), 'type', 'within_cluster');
        
    else
        %% FULL MATRIX TEST: Cluster-based permutation on difference
        % Mean difference across subjects (for descriptives)
        mean_diff_matrix = mean(diff_acc, 3);
        overall_mean_diff = mean(mean_diff_matrix, 'all');
        
        fprintf('  Mean difference (ref - other): %.4f\n', overall_mean_diff);
        
        % t-test on differences (testing if diff > 0, i.e., ref > other)
        obs_mean = mean(diff_acc, 3);
        obs_std = std(diff_acc, 0, 3);
        obs_t = obs_mean ./ (obs_std / sqrt(nSubj_paired));
        obs_t(~isfinite(obs_t)) = 0;
        
        % Find clusters above threshold
        obs_mask = obs_t > opts.t_thresh;
        cc_obs = bwconncomp(obs_mask, 8);
        
        if cc_obs.NumObjects > 0
            cluster_tsums = cellfun(@(x) sum(obs_t(x)), cc_obs.PixelIdxList);
            [obs_max_cluster, max_cluster_idx] = max(cluster_tsums);
            
            % Get cluster extent info
            [rows, cols] = ind2sub(size(obs_mask), cc_obs.PixelIdxList{max_cluster_idx});
            cluster_info = struct();
            cluster_info.tsum = obs_max_cluster;
            cluster_info.size = numel(cc_obs.PixelIdxList{max_cluster_idx});
            cluster_info.time_fb_range = [time_fb(min(rows)), time_fb(max(rows))];
            cluster_info.time_cue_range = [time_cue(min(cols)), time_cue(max(cols))];
            cluster_info.n_clusters = cc_obs.NumObjects;
            cluster_info.all_tsums = cluster_tsums;
            cluster_info.type = 'cluster_based';
            
            fprintf('  Observed: %d clusters, max cluster stat = %.2f\n', ...
                cc_obs.NumObjects, obs_max_cluster);
            fprintf('    Largest cluster: feedback [%.3f - %.3f]s, cue [%.3f - %.3f]s\n', ...
                cluster_info.time_fb_range(1), cluster_info.time_fb_range(2), ...
                cluster_info.time_cue_range(1), cluster_info.time_cue_range(2));
        else
            obs_max_cluster = 0;
            cluster_info = struct('tsum', 0, 'size', 0, 'n_clusters', 0, 'type', 'cluster_based');
            fprintf('  Observed: No clusters above threshold.\n');
        end
    end
    
    observed_clusters{idx} = cluster_info;
    
    %% Sign-flip permutation test
    fprintf('  Running %d permutations...\n', opts.nPerm);
    
    null_stat = zeros(opts.nPerm, 1);
    
    parfor perm = 1:opts.nPerm
        % Random sign flips for each subject
        flips = (randi([0 1], [1, 1, nSubj_paired]) * 2 - 1);
        
        % Apply sign flips to difference data
        perm_diff = diff_acc .* flips;
        
        if use_within_cluster
            % WITHIN-CLUSTER TEST: Compute t-stat on mean within cluster
            perm_subj_diff = zeros(nSubj_paired, 1);
            for s = 1:nSubj_paired
                perm_subj_matrix = perm_diff(:,:,s);
                perm_subj_diff(s) = mean(perm_subj_matrix(ref_cluster_mask));
            end
            perm_mean_diff = mean(perm_subj_diff);
            perm_std_diff = std(perm_subj_diff);
            null_stat(perm) = perm_mean_diff / (perm_std_diff / sqrt(nSubj_paired));
        else
            % FULL MATRIX TEST: Cluster-based
            perm_mean = mean(perm_diff, 3);
            perm_std = std(perm_diff, 0, 3);
            perm_t = perm_mean ./ (perm_std / sqrt(nSubj_paired));
            perm_t(~isfinite(perm_t)) = 0;
            
            % Find clusters
            perm_mask = perm_t > opts.t_thresh;
            cc_perm = bwconncomp(perm_mask, 8);
            
            if cc_perm.NumObjects > 0
                perm_tsums = cellfun(@(x) sum(perm_t(x)), cc_perm.PixelIdxList);
                null_stat(perm) = max(perm_tsums);
            else
                null_stat(perm) = 0;
            end
        end
    end
    
    null_distributions{idx} = null_stat;
    
    % Compute p-value (one-tailed: is observed >= null?)
    cluster_p = mean(null_stat >= obs_max_cluster);
    
    fprintf('  Cluster p-value: %.4f\n', cluster_p);
    
    if cluster_p < opts.alpha
        fprintf('  ** SIGNIFICANT at alpha = %.2f **\n', opts.alpha);
    end
    
    % Store results
    pairwise.ROI{idx} = ROIs{r};
    pairwise.nSubj_paired(idx) = nSubj_paired;
    pairwise.observed_cluster_stat(idx) = obs_max_cluster;
    pairwise.cluster_p(idx) = cluster_p;
    pairwise.mean_diff(idx) = overall_mean_diff;
    pairwise.sig(idx) = cluster_p < opts.alpha;
    
    idx = idx + 1;
end

%% Apply multiple comparison correction
n_comparisons = nROIs - 1;
pairwise.p_bonferroni = min(pairwise.cluster_p * n_comparisons, 1);
pairwise.sig_bonferroni = pairwise.p_bonferroni < opts.alpha;

% FDR correction (Benjamini-Hochberg)
valid_p = ~isnan(pairwise.cluster_p);
if any(valid_p)
    [sorted_p, sort_idx] = sort(pairwise.cluster_p(valid_p));
    m = sum(valid_p);
    fdr_thresh = (1:m)' / m * opts.alpha;
    sig_fdr = sorted_p <= fdr_thresh;
    max_k = find(sig_fdr, 1, 'last');
    if isempty(max_k)
        sig_fdr_final = false(m, 1);
    else
        sig_fdr_final = false(m, 1);
        sig_fdr_final(1:max_k) = true;
    end
    % Unsort
    temp_sig = false(sum(valid_p), 1);
    temp_sig(sort_idx) = sig_fdr_final;
    pairwise.sig_fdr = false(height(pairwise), 1);
    pairwise.sig_fdr(valid_p) = temp_sig;
else
    pairwise.sig_fdr = false(height(pairwise), 1);
end

%% Summary
fprintf('\n=== SUMMARY ===\n');
fprintf('Reference ROI: %s\n', opts.reference_roi);
fprintf('\n%-8s  %6s  %12s  %10s  %12s  %8s\n', ...
    'ROI', 'N', 'ClusterStat', 'p-value', 'p-Bonferroni', 'Sig');
fprintf('%s\n', repmat('-', 1, 65));

for i = 1:height(pairwise)
    sig_str = '';
    if pairwise.sig_bonferroni(i)
        sig_str = '**';
    elseif pairwise.sig(i)
        sig_str = '*';
    end
    fprintf('%-8s  %6d  %12.2f  %10.4f  %12.4f  %8s\n', ...
        pairwise.ROI{i}, pairwise.nSubj_paired(i), ...
        pairwise.observed_cluster_stat(i), pairwise.cluster_p(i), ...
        pairwise.p_bonferroni(i), sig_str);
end

fprintf('\n* p < %.2f (uncorrected)\n', opts.alpha);
fprintf('** p < %.2f (Bonferroni-corrected)\n', opts.alpha);

%% Compile results
results = struct();
results.pairwise = pairwise;
results.diff_matrices = diff_matrices;
results.observed_clusters = observed_clusters;
results.null_distributions = null_distributions;
results.time_fb = time_fb;
results.time_cue = time_cue;
results.ROIs = ROIs;
results.reference_roi = opts.reference_roi;
results.params = struct('nPerm', opts.nPerm, 't_thresh', opts.t_thresh, ...
    'alpha', opts.alpha);

%% --- PLOT AESTHETICS CONFIGURATION ---
plot_cfg = struct();
plot_cfg.font_name = 'Arial';
plot_cfg.font_size_label = 12;
plot_cfg.font_size_tick = 10;
plot_cfg.tick_dir = 'out';
plot_cfg.line_width = 1;
plot_cfg.tick_length = [0.025 0.025];

%% Visualization
fig_heatmap = figure('Position', [100 100 1200 400]);
set(fig_heatmap, 'Color', 'w');
set(fig_heatmap, 'Renderer', 'painters');

n_plots = sum(~isnan(pairwise.cluster_p));
plot_idx = 1;

for i = 1:height(pairwise)
    if isnan(pairwise.cluster_p(i))
        continue;
    end
    
    subplot(1, n_plots, plot_idx);
    
    % Plot mean difference matrix
    mean_diff_mat = mean(diff_matrices{i}, 3);
    
    imagesc(time_cue, time_fb, mean_diff_mat);
    axis xy;
    colorbar;
    caxis([-0.05 0.05]);
    colormap(redblue(256));
    
    xlabel('Cue time (s)');
    ylabel('Feedback time (s)');
    title(sprintf('%s - %s\np = %.4f', opts.reference_roi, pairwise.ROI{i}, ...
        pairwise.cluster_p(i)));
    
    % Overlay significant cluster contour or reference cluster
    if ~isempty(observed_clusters{i})
        hold on;
        if isfield(observed_clusters{i}, 'type') && strcmp(observed_clusters{i}.type, 'within_cluster')
            % For within-cluster test, overlay the reference cluster mask
            if ~isempty(ref_cluster_mask)
                contour(time_cue, time_fb, double(ref_cluster_mask), [0.5 0.5], 'k', 'LineWidth', 2);
            end
        elseif isfield(observed_clusters{i}, 'n_clusters') && observed_clusters{i}.n_clusters > 0
            % For cluster-based test, overlay discovered clusters
            obs_mean = mean(diff_matrices{i}, 3);
            obs_std = std(diff_matrices{i}, 0, 3);
            obs_t = obs_mean ./ (obs_std / sqrt(pairwise.nSubj_paired(i)));
            obs_t(~isfinite(obs_t)) = 0;
            sig_mask = obs_t > opts.t_thresh;
            contour(time_cue, time_fb, double(sig_mask), [0.5 0.5], 'k', 'LineWidth', 2);
        end
        hold off;
    end
    
    plot_idx = plot_idx + 1;
end

sgtitle(sprintf('Cross-Decoding Difference: %s vs Other Regions', opts.reference_roi));

%% Bar plot of regional comparisons
fig_barplot = figure('Position', [100 550 800 450]);

% Get valid comparisons
valid_idx = ~isnan(pairwise.cluster_p);
n_valid = sum(valid_idx);

if n_valid > 0
    % Compute mean difference and SEM for each comparison
    mean_diffs = zeros(n_valid, 1);
    sem_diffs = zeros(n_valid, 1);
    bar_labels = cell(n_valid, 1);
    p_vals = zeros(n_valid, 1);
    
    valid_rows = find(valid_idx);
    for k = 1:n_valid
        row_idx = valid_rows(k);
        
        % Get subject-level differences
        if use_within_cluster && ~isempty(ref_cluster_mask)
            % Within-cluster: mean diff within cluster for each subject
            diff_mat = diff_matrices{row_idx};
            nSubj = size(diff_mat, 3);
            subj_diffs = zeros(nSubj, 1);
            for s = 1:nSubj
                subj_diffs(s) = mean(diff_mat(:,:,s) .* ref_cluster_mask, 'all', 'omitnan') / mean(ref_cluster_mask(:));
            end
        else
            % Full matrix: mean diff across all time points
            diff_mat = diff_matrices{row_idx};
            nSubj = size(diff_mat, 3);
            subj_diffs = squeeze(mean(diff_mat, [1 2]));
        end
        
        mean_diffs(k) = mean(subj_diffs);
        sem_diffs(k) = std(subj_diffs) / sqrt(nSubj);
        bar_labels{k} = pairwise.ROI{row_idx};
        p_vals(k) = pairwise.cluster_p(row_idx);
    end
    
    % Create bar plot
    bar_h = bar(1:n_valid, mean_diffs, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1);
    hold on;
    
    % Add error bars
    errorbar(1:n_valid, mean_diffs, sem_diffs, 'k', 'LineStyle', 'none', 'LineWidth', 1.5, 'CapSize', 8);
    
    % Add zero line
    yline(0, 'k--', 'LineWidth', 1);
    
    % Add significance asterisks
    max_y = max(mean_diffs + sem_diffs);
    min_y = min(mean_diffs - sem_diffs);
    y_range = max_y - min_y;
    
    for k = 1:n_valid
        if p_vals(k) < 0.001
            sig_str = '***';
        elseif p_vals(k) < 0.01
            sig_str = '**';
        elseif p_vals(k) < 0.05
            sig_str = '*';
        else
            sig_str = 'n.s.';
        end
        
        % Position asterisks above/below bar depending on sign
        if mean_diffs(k) >= 0
            y_pos = mean_diffs(k) + sem_diffs(k) + 0.05 * y_range;
        else
            y_pos = mean_diffs(k) - sem_diffs(k) - 0.08 * y_range;
        end
        
        text(k, y_pos, sig_str, 'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
    
    hold off;
    
    % Labels and formatting
    set(gca, 'XTick', 1:n_valid, 'XTickLabel', bar_labels, 'FontSize', 11);
    xlabel('Comparison Region', 'FontSize', 12);
    ylabel(sprintf('Accuracy Difference (%s - Other)', opts.reference_roi), 'FontSize', 12);
    title(sprintf('Regional Comparison: %s vs Other Regions', opts.reference_roi), 'FontSize', 14);
    
    % Adjust y-axis limits
    ylim_padding = 0.2 * y_range;
    ylim([min_y - ylim_padding, max_y + ylim_padding]);
    
    % Add legend for significance
    text(0.98, 0.02, {'* p < 0.05', '** p < 0.01', '*** p < 0.001'}, ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'bottom', 'FontSize', 9, 'BackgroundColor', 'w');
    
    box on;
    grid on;
    set(gca, 'GridAlpha', 0.3);
end

%% ACC-dmPFC Correlation Analysis
fprintf('\n=== ACC-dmPFC Cluster Strength Correlation ===\n');

% Find indices for ACC and dmPFC
acc_idx = find(strcmp(ROIs, 'ACC'));
dmpfc_idx = find(strcmp(ROIs, 'dmPFC'));

% Check if both ROIs are available
if isempty(acc_idx) || isempty(dmpfc_idx)
    fprintf('ACC or dmPFC not in ROI list. Skipping correlation analysis.\n');
else
    % Load ACC and dmPFC data
    acc_fname = fullfile(output_dir, 'ACC_all_acc_subjects.mat');
    dmpfc_fname = fullfile(output_dir, 'dmPFC_all_acc_subjects.mat');
    
    if ~exist(acc_fname, 'file') || ~exist(dmpfc_fname, 'file')
        fprintf('ACC or dmPFC data files not found. Skipping correlation analysis.\n');
    else
        acc_data = load(acc_fname);
        dmpfc_data = load(dmpfc_fname);
        
        % Check for significant clusters
        if ~isfield(acc_data, 'sig_mask_final') || ~isfield(dmpfc_data, 'sig_mask_final')
            fprintf('Significant cluster masks not found. Skipping correlation analysis.\n');
        else
            acc_sig_mask = acc_data.sig_mask_final;
            dmpfc_sig_mask = dmpfc_data.sig_mask_final;
            
            if ~any(acc_sig_mask(:)) || ~any(dmpfc_sig_mask(:))
                fprintf('ACC or dmPFC has no significant clusters. Skipping correlation analysis.\n');
            else
                % Find subjects present in both ROIs
                if isfield(acc_data, 'subj_ids') && isfield(dmpfc_data, 'subj_ids')
                    acc_subj_ids = acc_data.subj_ids;
                    dmpfc_subj_ids = dmpfc_data.subj_ids;
                    [common_subjs, acc_idx_subj, dmpfc_idx_subj] = intersect(acc_subj_ids, dmpfc_subj_ids);
                else
                    % Fallback: assume aligned by index
                    n_common = min(size(acc_data.all_acc, 3), size(dmpfc_data.all_acc, 3));
                    common_subjs = cell(n_common, 1);
                    for i = 1:n_common
                        common_subjs{i} = sprintf('Subj%d', i);
                    end
                    acc_idx_subj = 1:n_common;
                    dmpfc_idx_subj = 1:n_common;
                end
                
                if numel(common_subjs) < 3
                    fprintf('Less than 3 subjects have both ACC and dmPFC data. Skipping correlation.\n');
                else
                    fprintf('Found %d subjects with both ACC and dmPFC data\n', numel(common_subjs));
                    
                    % Compute cluster strength for each subject in each ROI
                    acc_strength = zeros(numel(common_subjs), 1);
                    dmpfc_strength = zeros(numel(common_subjs), 1);
                    chance = 1/3;
                    
                    for i = 1:numel(common_subjs)
                        % ACC cluster strength (mean accuracy within significant cluster)
                        acc_subj_acc = acc_data.all_acc(:,:,acc_idx_subj(i));
                        acc_strength(i) = mean(acc_subj_acc(acc_sig_mask)) - chance;  % Effect size
                        
                        % dmPFC cluster strength
                        dmpfc_subj_acc = dmpfc_data.all_acc(:,:,dmpfc_idx_subj(i));
                        dmpfc_strength(i) = mean(dmpfc_subj_acc(dmpfc_sig_mask)) - chance;  % Effect size
                    end
                    
                    % Compute correlation
                    [r, p] = corr(acc_strength, dmpfc_strength, 'Type', 'Pearson');
                    fprintf('  Pearson r = %.4f, p = %.4f\n', r, p);
                    fprintf('  N = %d subjects\n', numel(common_subjs));
                    
                    % Create scatter plot
                    fig_scatter = figure('Position', [100 100 500 450]);
                    set(fig_scatter, 'Color', 'w');
                    set(fig_scatter, 'Renderer', 'painters');
                    
                    scatter(acc_strength, dmpfc_strength, 80, 'filled', 'MarkerFaceColor', [0.3 0.5 0.8], ...
                        'MarkerEdgeColor', 'k', 'LineWidth', plot_cfg.line_width);
                    hold on;
                    
                    % Add regression line if correlation is significant
                    if p < 0.05
                        % Fit linear regression
                        p_fit = polyfit(acc_strength, dmpfc_strength, 1);
                        x_fit = linspace(min(acc_strength), max(acc_strength), 100);
                        y_fit = polyval(p_fit, x_fit);
                        plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
                        
                        % Add text annotation with stats
                        text(0.05, 0.95, sprintf('r = %.3f\np = %.3f', r, p), ...
                            'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
                            'FontSize', plot_cfg.font_size_tick, 'VerticalAlignment', 'top', ...
                            'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', plot_cfg.line_width);
                    else
                        % Still show correlation even if not significant
                        text(0.05, 0.95, sprintf('r = %.3f\np = %.3f (n.s.)', r, p), ...
                            'Units', 'normalized', 'FontName', plot_cfg.font_name, ...
                            'FontSize', plot_cfg.font_size_tick, 'VerticalAlignment', 'top', ...
                            'BackgroundColor', 'w', 'EdgeColor', 'k', 'LineWidth', plot_cfg.line_width);
                    end
                    
                    % Axis labels and styling
                    xlabel('ACC Reactivation Strength', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
                    ylabel('dmPFC Reactivation Strength', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label);
                    title('Cross-ROI Reactivation Correlation', 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_label + 2);
                    
                    set(gca, 'FontName', plot_cfg.font_name, 'FontSize', plot_cfg.font_size_tick);
                    set(gca, 'TickDir', plot_cfg.tick_dir);
                    set(gca, 'LineWidth', plot_cfg.line_width);
                    set(gca, 'TickLength', plot_cfg.tick_length);
                    box off;
                    grid on;
                    grid minor;
                    set(gca, 'GridAlpha', 0.3);
                    hold off;
                    
                    % Save figure
                    scatter_file = fullfile(output_dir, 'ACC_dmPFC_correlation.pdf');
                    print(fig_scatter, scatter_file, '-dpdf', '-r300');
                    fprintf('Correlation plot saved to: %s\n', scatter_file);
                    
                    % Save correlation data
                    corr_table = table(common_subjs(:), acc_strength, dmpfc_strength, ...
                        'VariableNames', {'Subject', 'ACC_ReactivationStrength', 'dmPFC_ReactivationStrength'});
                    writetable(corr_table, fullfile(output_dir, 'ACC_dmPFC_correlation_data.csv'));
                    fprintf('Correlation data saved to: %s\n', fullfile(output_dir, 'ACC_dmPFC_correlation_data.csv'));
                end
            end
        end
    end
end

%% Save results
if opts.save_results
    % Save bar plot figure
    barplot_file = fullfile(output_dir, sprintf('%s_regional_comparison_barplot.png', opts.reference_roi));
    saveas(fig_barplot, barplot_file);
    fprintf('\nBar plot saved to: %s\n', barplot_file);
    
    % Save heatmap figure
    fig_file = fullfile(output_dir, sprintf('%s_regional_comparison_heatmaps.png', opts.reference_roi));
    saveas(fig_heatmap, fig_file);
    fprintf('Heatmaps saved to: %s\n', fig_file);
    
    % Save .mat
    save_file = fullfile(output_dir, sprintf('%s_regional_comparison_permutation.mat', opts.reference_roi));
    save(save_file, '-struct', 'results', '-v7.3');
    fprintf('Results saved to: %s\n', save_file);
    
    % Save text report
    txt_file = fullfile(output_dir, sprintf('%s_regional_comparison_permutation_report.txt', opts.reference_roi));
    fid = fopen(txt_file, 'w');
    
    fprintf(fid, 'REGIONAL COMPARISON: SIGN-FLIP CLUSTER PERMUTATION TEST\n');
    fprintf(fid, '========================================================\n\n');
    fprintf(fid, 'Reference ROI: %s\n', opts.reference_roi);
    fprintf(fid, 'Number of permutations: %d\n', opts.nPerm);
    fprintf(fid, 'Cluster-forming threshold: t = %.2f\n', opts.t_thresh);
    fprintf(fid, 'Alpha level: %.4f\n\n', opts.alpha);
    
    fprintf(fid, 'METHOD\n');
    fprintf(fid, '------\n');
    fprintf(fid, 'For each comparison, the difference in accuracy (ref - other) is computed\n');
    fprintf(fid, 'for each subject at each time point. A one-sample t-test against zero is\n');
    fprintf(fid, 'performed at each time point, and clusters of above-threshold t-values are\n');
    fprintf(fid, 'identified. The cluster statistic is the sum of t-values within the cluster.\n');
    fprintf(fid, 'A null distribution is generated by randomly flipping the sign of each\n');
    fprintf(fid, 'subject''s difference scores and recomputing the max cluster statistic.\n');
    fprintf(fid, 'The p-value is the proportion of null cluster stats >= observed.\n\n');
    
    fprintf(fid, 'RESULTS\n');
    fprintf(fid, '-------\n');
    fprintf(fid, '%-8s  %6s  %12s  %10s  %12s\n', ...
        'ROI', 'N', 'ClusterStat', 'p-value', 'p-Bonferroni');
    
    for i = 1:height(pairwise)
        fprintf(fid, '%-8s  %6d  %12.2f  %10.4f  %12.4f\n', ...
            pairwise.ROI{i}, pairwise.nSubj_paired(i), ...
            pairwise.observed_cluster_stat(i), pairwise.cluster_p(i), ...
            pairwise.p_bonferroni(i));
    end
    
    fprintf(fid, '\nSIGNIFICANT COMPARISONS (Bonferroni-corrected):\n');
    sig_rois = pairwise.ROI(pairwise.sig_bonferroni);
    if isempty(sig_rois) || all(cellfun(@isempty, sig_rois))
        fprintf(fid, '  None\n');
    else
        for i = 1:numel(sig_rois)
            if ~isempty(sig_rois{i})
                fprintf(fid, '  %s > %s\n', opts.reference_roi, sig_rois{i});
            end
        end
    end
    
    fclose(fid);
    fprintf('Report saved to: %s\n', txt_file);
end

end

%% Helper function for colormap
function cmap = redblue(n)
    if nargin < 1
        n = 256;
    end
    
    % Red-white-blue colormap
    top = [0.7 0 0];
    middle = [1 1 1];
    bottom = [0 0 0.7];
    
    half = floor(n/2);
    
    r = [linspace(bottom(1), middle(1), half), linspace(middle(1), top(1), n-half)];
    g = [linspace(bottom(2), middle(2), half), linspace(middle(2), top(2), n-half)];
    b = [linspace(bottom(3), middle(3), half), linspace(middle(3), top(3), n-half)];
    
    cmap = [r', g', b'];
end

