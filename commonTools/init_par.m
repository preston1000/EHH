function parameters = init_par(config_file)
% This is to read algorithm parameters and check the validity of these
% parameters.
% 
% Input:
%       config_file: configuration file name, or absolute path

    fprintf('Read the config file: %s\n', config_file)
    parameters = ini2struct(config_file);
    fields = fieldnames(parameters);
    for i = 1:length(fields)
        if strcmpi(fields{i}, 'structure_candidate')
            parameters.(fields{i}) = processCell(parameters.(fields{i}));
        elseif strcmpi(fields{i}, 'dataProcessFunction')
            continue;
        else
            parameters.(fields{i}) = str2num(parameters.(fields{i})); %#ok<ST2NM>
        end
    end
    fprintf('Succeeded.\n')
end

function values = processCell(text)
    values = cell(100, 1);
    counter = 1;
    while 1
        [one, two] = strtok(text, ',');
        
        values{counter} = str2num(one); %#ok<ST2NM>
        counter = counter + 1;
        if isempty(two) 
            break
        else
            text = cleanValue(two);
        end
    end
    values = values(1:(counter - 1));
end

function res = cleanValue(s)
    res = strtrim(s);
    if strcmpi(res(1),',')
        res(1)=[];
    end
    res = strtrim(res);
end