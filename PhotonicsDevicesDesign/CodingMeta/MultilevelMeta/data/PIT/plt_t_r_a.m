clear all;

date_transmission_PIT_trans_simple_100_100_100=load('PIT_trans_simple_100_100_100.txt');
date1_transmission_PIT_trans_simple_100_100_100=date_transmission_PIT_trans_simple_100_100_100(:,1);
date2_transmission_PIT_trans_simple_100_100_100=date_transmission_PIT_trans_simple_100_100_100(:,2);
date_transmission_PIT_trans_simple_100_50_50=load('PIT_trans_simple_100_50_50.txt');
date1_transmission_PIT_trans_simple_100_50_50=date_transmission_PIT_trans_simple_100_50_50(:,1);
date2_transmission_PIT_trans_simple_100_50_50=date_transmission_PIT_trans_simple_100_50_50(:,2);
date_transmission_PIT_trans_simple_100_0_0=load('PIT_trans_simple_100_0_0.txt');
date1_transmission_PIT_trans_simple_100_0_0=date_transmission_PIT_trans_simple_100_0_0(:,1);
date2_transmission_PIT_trans_simple_100_0_0=date_transmission_PIT_trans_simple_100_0_0(:,2);
date_transmission_PIT_trans_simple_50_100_100=load('PIT_trans_simple_50_100_100.txt');
date1_transmission_PIT_trans_simple_50_100_100=date_transmission_PIT_trans_simple_50_100_100(:,1);
date2_transmission_PIT_trans_simple_50_100_100=date_transmission_PIT_trans_simple_50_100_100(:,2);
date_transmission_PIT_trans_simple_0_100_100=load('PIT_trans_simple_0_100_100.txt');
date1_transmission_PIT_trans_simple_0_100_100=date_transmission_PIT_trans_simple_0_100_100(:,1);
date2_transmission_PIT_trans_simple_0_100_100=date_transmission_PIT_trans_simple_0_100_100(:,2);

date_transmission_PIT_trans_complex_100_100_100_100=load('PIT_trans_complex_100_100_100_100.txt');
date1_transmission_PIT_trans_complex_100_100_100_100=date_transmission_PIT_trans_complex_100_100_100_100(:,1);
date2_transmission_PIT_trans_complex_100_100_100_100=date_transmission_PIT_trans_complex_100_100_100_100(:,2);
date_transmission_PIT_trans_complex_0_100_100_100=load('PIT_trans_complex_0_100_100_100.txt');
date1_transmission_PIT_trans_complex_0_100_100_100=date_transmission_PIT_trans_complex_0_100_100_100(:,1);
date2_transmission_PIT_trans_complex_0_100_100_100=date_transmission_PIT_trans_complex_0_100_100_100(:,2);
date_transmission_PIT_trans_complex_100_0_0_100=load('PIT_trans_complex_100_0_0_100.txt');
date1_transmission_PIT_trans_complex_100_0_0_100=date_transmission_PIT_trans_complex_100_0_0_100(:,1);
date2_transmission_PIT_trans_complex_100_0_0_100=date_transmission_PIT_trans_complex_100_0_0_100(:,2);
date_transmission_PIT_trans_complex_50_100_100_100=load('PIT_trans_complex_50_100_100_100.txt');
date1_transmission_PIT_trans_complex_50_100_100_100=date_transmission_PIT_trans_complex_50_100_100_100(:,1);
date2_transmission_PIT_trans_complex_50_100_100_100=date_transmission_PIT_trans_complex_50_100_100_100(:,2);
date_transmission_PIT_trans_complex_100_50_50_100=load('PIT_trans_complex_100_50_50_100.txt');
date1_transmission_PIT_trans_complex_100_50_50_100=date_transmission_PIT_trans_complex_100_50_50_100(:,1);
date2_transmission_PIT_trans_complex_100_50_50_100=date_transmission_PIT_trans_complex_100_50_50_100(:,2);
date_transmission_PIT_trans_complex_100_100_100_50=load('PIT_trans_complex_100_100_100_50.txt');
date1_transmission_PIT_trans_complex_100_100_100_50=date_transmission_PIT_trans_complex_100_100_100_50(:,1);
date2_transmission_PIT_trans_complex_100_100_100_50=date_transmission_PIT_trans_complex_100_100_100_50(:,2);

date_transmission_PIT_trans_complex_100_100_100_100_100=load('PIT_trans_complex_100_100_100_100_100.txt');
date1_transmission_PIT_trans_complex_100_100_100_100_100=date_transmission_PIT_trans_complex_100_100_100_100_100(:,1);
date2_transmission_PIT_trans_complex_100_100_100_100_100=date_transmission_PIT_trans_complex_100_100_100_100_100(:,2);
date_transmission_PIT_trans_complex_100_0_0_100_100=load('PIT_trans_complex_100_0_0_100_100.txt');
date1_transmission_PIT_trans_complex_100_0_0_100_100=date_transmission_PIT_trans_complex_100_0_0_100_100(:,1);
date2_transmission_PIT_trans_complex_100_0_0_100_100=date_transmission_PIT_trans_complex_100_0_0_100_100(:,2);
date_transmission_PIT_trans_complex_100_0_0_100_0=load('PIT_trans_complex_100_0_0_100_0.txt');
date1_transmission_PIT_trans_complex_100_0_0_100_0=date_transmission_PIT_trans_complex_100_0_0_100_0(:,1);
date2_transmission_PIT_trans_complex_100_0_0_100_0=date_transmission_PIT_trans_complex_100_0_0_100_0(:,2);
date_transmission_PIT_trans_complex_100_100_0_100_100=load('PIT_trans_complex_100_100_0_100_100.txt');
date1_transmission_PIT_trans_complex_100_100_0_100_100=date_transmission_PIT_trans_complex_100_100_0_100_100(:,1);
date2_transmission_PIT_trans_complex_100_100_0_100_100=date_transmission_PIT_trans_complex_100_100_0_100_100(:,2);
date_transmission_PIT_trans_complex_0_100_100_0_100=load('PIT_trans_complex_0_100_100_0_100.txt');
date1_transmission_PIT_trans_complex_0_100_100_0_100=date_transmission_PIT_trans_complex_0_100_100_0_100(:,1);
date2_transmission_PIT_trans_complex_0_100_100_0_100=date_transmission_PIT_trans_complex_0_100_100_0_100(:,2);
date_transmission_PIT_trans_complex_0_0_0_0_100=load('PIT_trans_complex_0_0_0_0_100.txt');
date1_transmission_PIT_trans_complex_0_0_0_0_100=date_transmission_PIT_trans_complex_0_0_0_0_100(:,1);
date2_transmission_PIT_trans_complex_0_0_0_0_100=date_transmission_PIT_trans_complex_0_0_0_0_100(:,2);

figure(1)
plot(date1_transmission_PIT_trans_simple_100_100_100,date2_transmission_PIT_trans_simple_100_100_100, 'r', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_simple_100_50_50,date2_transmission_PIT_trans_simple_100_50_50, 'b', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_simple_100_0_0,date2_transmission_PIT_trans_simple_100_0_0, 'k', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_simple_50_100_100,date2_transmission_PIT_trans_simple_50_100_100, 'm', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_simple_0_100_100,date2_transmission_PIT_trans_simple_0_100_100, 'c', 'LineWidth', 2);
h=gca;
set(h, 'FontName', 'Times New Roman', 'FontSize', 25);
axis([500, 2000, 0, 1]);

figure(2)
plot(date1_transmission_PIT_trans_complex_100_100_100_100,date2_transmission_PIT_trans_complex_100_100_100_100, 'r', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_0_100_100_100,date2_transmission_PIT_trans_complex_0_100_100_100, 'b', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_0_0_100,date2_transmission_PIT_trans_complex_100_0_0_100, 'k', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_50_100_100_100,date2_transmission_PIT_trans_complex_50_100_100_100, 'g', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_50_50_100,date2_transmission_PIT_trans_complex_100_50_50_100, 'm', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_100_100_50,date2_transmission_PIT_trans_complex_100_100_100_50, 'c', 'LineWidth', 2);
h=gca;
set(h, 'FontName', 'Times New Roman', 'FontSize', 25);
axis([500, 2000, 0, 1]);

figure(3)
plot(date1_transmission_PIT_trans_complex_100_100_100_100_100,date2_transmission_PIT_trans_complex_100_100_100_100_100, 'r', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_0_0_100_100,date2_transmission_PIT_trans_complex_100_0_0_100_100, 'b--', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_0_0_100_0,date2_transmission_PIT_trans_complex_100_0_0_100_0, 'k', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_100_100_0_100_100,date2_transmission_PIT_trans_complex_100_100_0_100_100, 'g', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_0_100_100_0_100,date2_transmission_PIT_trans_complex_0_100_100_0_100, 'm--', 'LineWidth', 2);
hold on
plot(date1_transmission_PIT_trans_complex_0_0_0_0_100,date2_transmission_PIT_trans_complex_0_0_0_0_100, 'c', 'LineWidth', 2);
h=gca;
set(h, 'FontName', 'Times New Roman', 'FontSize', 25);
axis([500, 2000, 0, 1]);