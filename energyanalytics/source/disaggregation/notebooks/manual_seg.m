
start_time = datetime(2016,4,1);
end_time = datetime(2016,7, 31);
current_time = start_time;

%%
while current_time <= end_time
    [t, dat, t_str] = load_dat(current_time.Month,current_time.Day,...
        0,23,'../new_data/IHG');
    
    close all
    [ fig, figLocation ] = createfig3('figsize', [25,8])
    subplot(2,1,1)
    plot(t, dat, 'k.-')
    
    xlim([0, 24])
    ylim([0, 500])
    set(gca, 'xtick', 0:24)
    grid on
    hold on

    
    subplot(2,1,2)
    plot(t, dat, 'k.-')
    
    xlim([0, 24])
    ylim([0, 500])
    set(gca, 'xtick', 0:24)
    grid on
    hold on
    
    cp_pos_list = [];
    cp_pos_index_list = [];
    
    while 1
        subplot(2,1,1)
        [click_pos, ~, button] = ginput(1);
        if click_pos
            click_pos = click_pos(1);
            
            subplot(2,1,1)
            h1 = plot(click_pos-.5+[0,0], [0,500], 'k-');
            h2 = plot(click_pos+.5+[0,0], [0,500], 'k-');
            
            subplot(2,1,2)
            xlim(click_pos+[-.5, .5])
            
            cp_pos = ginput(1);
            cp_pos = cp_pos(1);
            [~, cp_pos_index] = min(abs(t - cp_pos));
            cp_pos = t(cp_pos_index);
            if ismember(cp_pos, cp_pos_list)
                cp_pos_list (find(cp_pos_list == cp_pos)) = [];
                cp_pos_index_list (find(cp_pos_list == cp_pos)) = [];
                
                subplot(2,1,2)
                plot([cp_pos, cp_pos], [0,500], 'b--');
                
                subplot(2,1,1)
                plot([cp_pos, cp_pos], [0,500], 'b--');
            else
                cp_pos_list(end+1) = cp_pos
                cp_pos_index_list(end+1) = cp_pos_index
                
                delete(h1)
                delete(h2)
                
                subplot(2,1,2)
                plot([cp_pos, cp_pos], [0,500], 'r--');
                
                subplot(2,1,1)
                plot([cp_pos, cp_pos], [0,500], 'r--');
            end
            
        else
            break
        end
    end
    
    fid = fopen(fullfile('../results', sprintf('%d_%d_manual_seg.txt', current_time.Month,current_time.Day)), 'w');
    for i = 1:length(cp_pos_index_list)
        fprintf(fid, '%f\t%s\n', cp_pos_index_list(i), t_str{cp_pos_index_list(i)});
    end
    fclose(fid);
    
    export_fig( fig, fullfile('../results', sprintf('%d_%d_manual_seg.jpg', current_time.Month,current_time.Day)) )
    current_time = current_time+1;
end

