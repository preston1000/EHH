function parameters = init_par(config_file)
    fprintf('开始读取配置文件：%s\n', config_file)
    parameters = ini2struct(config_file);
    fields = fieldnames(parameters);
    for i = 1:length(fields)
        parameters.(fields{i}) = str2num(parameters.(fields{i})); %#ok<ST2NM>
    end
    fprintf('配置文件读取完成\n')