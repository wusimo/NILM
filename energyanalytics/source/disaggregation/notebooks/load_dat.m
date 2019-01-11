function [t, dat, t_str] = load_dat(m, d, h_start, h_end, folder)
t = [];
dat = [];
t_str = {};
for h = h_start:h_end
    [t_tmp, dat_tmp, t_str_tmp] = load_dat_helper( m, d, h, folder );
    t = [t, t_tmp+h];
    dat = [dat, dat_tmp];
    t_str = {t_str{:}, t_str_tmp{:}};
end

end


function [ t, dat, t_str ] = load_dat_helper( m, d, hour, folder )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

filename = [num2str(m), '-', num2str(d), '-', num2str(hour), '.csv'];
filepath = fullfile(folder, filename);

fid = fopen(filepath, 'r');
i_line = 1;
t = nan(1,240);
dat = nan(1,240);
t_str = cell(1,240);
while ~feof(fid)
    line = fgets(fid);
    tokens = regexp(line, ',', 'split');
%     tokens{1}
%     
%     tmp = datetime(tokens{1}(1:end-6));
%     time_tmp = tmp.Hour - hour + tmp.Minute/60 + tmp.Second/3600;
    
    tmp = regexp(tokens{1}(1:end-6), ':', 'split');
    time_tmp = str2num(tmp{2})/60 + str2num(tmp{3})/3600;
    
    dat_tmp = str2num(tokens{2})/3;
    
    t_str{i_line} = tokens{1};
    t(i_line) = time_tmp;
    dat(i_line) = dat_tmp;
    
    i_line = i_line+1;
end
fclose (fid);

end

