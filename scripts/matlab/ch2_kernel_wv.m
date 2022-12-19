clear all
close all
clc

rng(1)


COLOR_BLUE = '#1F77B4';
COLOR_RED = '#D62728';
COLOR_ORANGE = '#FF7F0E';


f = @(x) x.^2 + 2*x;

a = 1; b = 1.5; c = -0.8;
kernel = @(x1,x2) tanh(a^2 + b^2*(x1-c)*(x2-c)');

n_samp = 120;
xmin = -1; xmax = 1;
xx = linspace(xmin,xmax,n_samp)';

plot(xx,kernel(xx,0),'color',COLOR_BLUE);

% custom formatting
title('sigmoid','Interpreter','latex')

yl = ylabel('$k(x,0)$','Interpreter','latex');
ylim([0.6 1.03]); yticks([])
ref = ylim;

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

exportgraphics(gcf, '../images/chap2_kernel_sg.pdf','ContentType', 'vector');

%%