function run_models(path_to_matcode, model_names, seed)

    arguments
        path_to_matcode = fileparts(mfilename('fullpath'))
        model_names = string.empty
        seed (1,1) double = 985
    end

    path_to_matcode = char(path_to_matcode);
    addpath(path_to_matcode);

    if isempty(model_names)
        model_names = find_example_models(path_to_matcode);
    else
        model_names = string(model_names);
    end

    log_file = fullfile(path_to_matcode, 'batch_run_log.txt');
    fid = fopen(log_file, 'w');

    if fid == -1
        error('Could not open log file for writing: %s', log_file);
    end

    cleanup = onCleanup(@() fclose(fid));

    log_message(fid, 'Found %d models\n', numel(model_names));

    for i = 1:numel(model_names)
        model_name = char(model_names(i));
    
        log_message(fid, '\n==== Running %s (%d/%d) ====\n', ...
            model_name, i, numel(model_names));
    
        try
            run_single_model(model_name, path_to_matcode, seed);
        catch ME
            log_message(fid, 'FAILED: %s\n%s\n', model_name, ME.message);
            log_message(fid, '%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
            continue
        end
    
        log_message(fid, 'FINISHED: %s\n', model_name);
        drawnow;
    end
    log_message(fid, '\nBatch complete. Log saved to:\n%s\n', log_file);
end

function model_names = find_example_models(path_to_matcode)
    models_root = fullfile(path_to_matcode, 'inputs', 'example_inputs');
    d = dir(models_root);

    model_names = string({d([d.isdir]).name});
    model_names = model_names(~ismember(model_names, [".", ".."]));
    model_names = model_names(contains(model_names, "_IM"));
end

function log_message(fid, varargin)
    fprintf(2, varargin{:});
    fprintf(fid, varargin{:});
end


% e.g., if one model only:
% run_example_models('C:\path\to\PBEE-Recovery', "2009_S5a_2_IM4")