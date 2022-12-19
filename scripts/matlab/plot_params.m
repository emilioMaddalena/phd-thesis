
close all
figure

fill([xx' fliplr(xx')], [UB fliplr(LB)], 'k', 'FaceColor', '#0072BD', 'EdgeColor', '#0072BD', 'facealpha', 0.1, 'linewidth', 1); hold on
plot(xx,f(xx), 'k--', 'linewidth', 1.5); hold on;

plot(X(1),y(1)-0.05,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
plot(X(2),y(2)-0.08,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')
plot(X(3),y(3)+0.2,'o','markersize',6,'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g')

xlim([xmin xmax]); ylim([-2 2]);
xticks([]); yticks([]);
set(gcf,'Color','w','Position',[10 10 350 250]);
set(gca,'box','off','xcolor','w','ycolor','w')

hline_loc = -1.9;
yline(hline_loc ,'linewidth',1.5); hold on
plot([X(2) X(2)],[-2 1.7],'k--','linewidth',1)
plot([X(3) X(3)],[-2 1.7],'k--','linewidth',1)

text(-0.9,-0.7,'$f^\star$','fontsize',16,'Interpreter','latex');
text(-0.65,0.15,'$C(x)$','fontsize',16,'Interpreter','latex');
text(-0.6,-1.33,'$B(x)$','fontsize',16,'Interpreter','latex');
text(X(2)-0.035,hline_loc-0.25,'$x_i$','fontsize',16,'Interpreter','latex');
text(X(3)-0.035,hline_loc-0.25,'$x_j$','fontsize',16,'Interpreter','latex');
text(X(2)+0.02,fX(2)-0.28,'$y_i$','fontsize',16,'Interpreter','latex');
text(X(3)+0.02,fX(3)-0.39,'$y_j$','fontsize',16,'Interpreter','latex');

obs = findall(gca,'Type','Line','Marker','o');
for i = 1:numel(obs), uistack(obs(i),'top'); end

exportgraphics(gcf, ['ex0_A.pdf'], 'ContentType', 'vector')