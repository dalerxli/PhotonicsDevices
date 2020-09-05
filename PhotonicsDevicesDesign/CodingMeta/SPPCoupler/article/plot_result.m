clear all;

date_two_layer_graphene_ribbons = load('3D_TE.txt');
date1_two_layer_graphene_ribbons=date_two_layer_graphene_ribbons(:,1);
date2_two_layer_graphene_ribbons=date_two_layer_graphene_ribbons(:,2);

figure(1)
plot(date1_two_layer_graphene_ribbons,date2_two_layer_graphene_ribbons+0.10,'r','LineWidth',3);
h=gca;
set(h,'FontName','Times New Roman','FontSize',25);
axis([1.45, 1.65, 0.75, 0.95]);