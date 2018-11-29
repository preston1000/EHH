function parameters = init_par(config_file)
    try
        parameters = ini2struct(config_file);
        fields = fieldnames(parameters);
        for i = 1:length(fields)
            parameters.(fields{i}) = str2num(parameters.(fields{i})); %#ok<ST2NM>
        end
    catch e
        parameters = e.identifier;
    end