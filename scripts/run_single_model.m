function run_single_model(model_name, path_to_matcode, seed)

    arguments
        model_name {mustBeTextScalar}
        path_to_matcode {mustBeTextScalar}
        seed (1,1) double = 985
    end

    model_name = char(model_name);
    path_to_matcode = char(path_to_matcode);

    input_root = fullfile(path_to_matcode, 'inputs', 'example_inputs');
    model_dir = fullfile(input_root, model_name);
    output_dir = fullfile(path_to_matcode, 'outputs', model_name);

    copy_input_scripts(path_to_matcode, model_dir);
    ensure_comp_population_csv(model_dir, model_name);

    start_dir = pwd;
    cleanup = onCleanup(@() cd(start_dir));
    cd(model_dir);

    % preserve runner variables so they do not get cleared 
    evalin('base', sprintf('model_name = ''%s'';', model_name));
    evalin('base', sprintf('seed = %g;', seed));
    evalin('base', sprintf('path_to_matcode = ''%s'';', path_to_matcode));
    
    evalin('base', sprintf('cd(''%s'');', model_dir));
    evalin('base', 'run(''optional_inputs.m'');');
    evalin('base', 'run(''build_inputs.m'');');
    
    cd(model_dir);
    
    sim_inputs_path = fullfile(model_dir, 'simulated_inputs.mat');
    if ~isfile(sim_inputs_path)
        error('Expected file not found: %s', sim_inputs_path);
    end

    load(sim_inputs_path);

    building_model = normalize_comps_story(building_model);

    systems = readtable(fullfile(path_to_matcode, 'static_tables', 'systems.csv'));
    subsystems = readtable(fullfile(path_to_matcode, 'static_tables', 'subsystems.csv'));
    impeding_factor_medians = readtable(fullfile(path_to_matcode, 'static_tables', 'impeding_factors.csv'));
    tmp_repair_class = readtable(fullfile(path_to_matcode, 'static_tables', 'temp_repair_class.csv'));

    [functionality, damage_consequences] = main_PBEErecovery( ...
        damage, damage_consequences, building_model, tenant_units, ...
        systems, subsystems, tmp_repair_class, impedance_options, ...
        impeding_factor_medians, repair_time_options, functionality, ...
        functionality_options);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    mat_file = fullfile(output_dir, 'recovery_outputs.mat');
    json_file = fullfile(output_dir, 'recovery_outputs_MATLAB.json');

    save(mat_file, 'functionality');
    write_json_output(mat_file, json_file);

    fprintf('Recovery assessment of model %s complete\n', model_name);
end

function copy_input_scripts(path_to_matcode, model_dir)
    input_maker_dir = fullfile(path_to_matcode, 'inputs', 'Inputs2Copy');

    copyfile(fullfile(input_maker_dir, 'build_inputs.m'), model_dir);
    copyfile(fullfile(input_maker_dir, 'optional_inputs.m'), model_dir);
end

function ensure_comp_population_csv(model_dir, model_name)
    csv_path = fullfile(model_dir, 'comp_population.csv');
    xlsx_path = fullfile(model_dir, 'comp_population.xlsx');
    xls_path = fullfile(model_dir, 'comp_population.xls');

    if isfile(csv_path)
        return
    end

    if isfile(xlsx_path)
        src = xlsx_path;
    elseif isfile(xls_path)
        src = xls_path;
    else
        error('Missing comp_population.csv, .xlsx, or .xls in: %s', model_dir);
    end

    T = readtable(src);

    for v = 1:width(T)
        if isstring(T{:, v}) || iscategorical(T{:, v})
            T{:, v} = cellstr(T{:, v});
        end
    end

    writetable(T, csv_path);
    fprintf('Created comp_population.csv for %s\n', model_name);
end

function building_model = normalize_comps_story(building_model)
    if ~isfield(building_model, 'comps') || ~isfield(building_model.comps, 'story')
        return
    end

    if istable(building_model.comps.story)
        building_model.comps.story = table2struct(building_model.comps.story);
    elseif iscell(building_model.comps.story)
        building_model.comps.story = [building_model.comps.story{:}];
    end
end

function write_json_output(mat_file, json_file)
    data = load(mat_file);
    fid = fopen(json_file, 'w');

    if fid == -1
        error('Could not open file for writing: %s', json_file);
    end

    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(data), 'char');
end