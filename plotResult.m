function plotResult( figID, y_ori, y_pre )
%PLOTRESULT plots the prediction performances

    figure(figID)
    clf
    plot(y_ori, 'k:','linewidth', 1.5)
    hold on
    plot(y_pre, 'linewidth', 1.5)
    xlabel('Times Instant', 'fontsize', 14)
    ylabel('The outputs', 'fontsize', 14)

end

