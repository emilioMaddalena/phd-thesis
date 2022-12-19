clear all
close all
clc

rng(1)


COLOR_BLUE = '#1F77B4';
COLOR_RED = '#D62728';
COLOR_ORANGE = '#FF7F0E';

kernel = @(x1,x2) x1 + x2;

n_samp = 100;
xmin = -4; xmax = 4;
xx = linspace(xmin,xmax,n_samp)';

plot(xx,kernel(xx,0),'color',COLOR_BLUE);

% custom formatting
title('linear','Interpreter','latex')

yl = ylabel('$k(x,0)$','Interpreter','latex');
ylim([-4 4]); yticks([])
ref = ylim

xl = xlabel('$x$','Interpreter','latex');
xl.Position = [0 ref(1)-(ref(2)-ref(1))/10 -1];
xticks(0)


% formatting
set(gcf,'Color','w','Position',[500 500 350 250]);
set(gca,'box','on','xcolor','k','ycolor','k',...
        'FontSize',16,...
        'TickLabelInterpreter','latex',...
        'Position', [0.18,0.11,0.7,0.83],...
        'PlotBoxAspectRatio',[0.6,0.45,1]);

my_lines = findall(gca,'Type','Line');
my_lines.LineWidth = 2;

exportgraphics(gcf, '../images/chap2_kernel_li.pdf','ContentType', 'vector');

%%