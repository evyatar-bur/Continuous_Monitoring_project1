function [score] = F1_score(prediction,Y_test)
% Compute weighted F1 score

% Compute f1 score for 0 class
tp_0 = sum((prediction == 0) & (Y_test == 0));
fp_0 = sum((prediction == 0) & (Y_test ~= 0));
fn_0 = sum((prediction ~= 0) & (Y_test == 0));

precision_0 = tp_0 / (tp_0 + fp_0);
recall_0 = tp_0 / (tp_0 + fn_0);
score_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0);

% Compute f1 score for 1 class
tp_1 = sum((prediction == 12 | prediction == 11) & (Y_test == 11 | Y_test == 12));
fp_1 = sum((prediction == 12 | prediction == 11) & (Y_test ~= 11 & Y_test ~= 12));
fn_1 = sum((prediction ~= 12 & prediction ~= 11) & (Y_test == 11 | Y_test == 12));

precision_1 = tp_1 / (tp_1 + fp_1);
recall_1 = tp_1 / (tp_1 + fn_1);
score_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1);

% Compute f1 score for 2 class
tp_2 = sum((prediction == 22 | prediction == 21) & (Y_test == 21 | Y_test == 22));
fp_2 = sum((prediction == 22 | prediction == 21) & (Y_test ~= 21 & Y_test ~= 22));
fn_2 = sum((prediction ~= 22 & prediction ~= 21) & (Y_test == 21 | Y_test == 22));

precision_2 = tp_2 / (tp_2 + fp_2);
recall_2 = tp_2 / (tp_2 + fn_2);
score_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2);

% Compute f1 score for 3 class
tp_3 = sum((prediction == 3) & (Y_test == 3));
fp_3 = sum((prediction == 3) & (Y_test ~= 3));
fn_3 = sum((prediction ~= 3) & (Y_test == 3));

precision_3 = tp_3 / (tp_3 + fp_3);
recall_3 = tp_3 / (tp_3 + fn_3);
score_3 = (2 * precision_3 * recall_3) / (precision_3 + recall_3);

% Compute f1 score for 4 class
tp_4 = sum((prediction == 4) & (Y_test == 4));
fp_4 = sum((prediction == 4) & (Y_test ~= 4));
fn_4 = sum((prediction ~= 4) & (Y_test == 4));

precision_4 = tp_4 / (tp_4 + fp_4);
recall_4 = tp_4 / (tp_4 + fn_4);
score_4 = (2 * precision_4 * recall_4) / (precision_4 + recall_4);

% Compute f1 score for 5 class
tp_5 = sum((prediction == 5) & (Y_test == 5));
fp_5 = sum((prediction == 5) & (Y_test ~= 5));
fn_5 = sum((prediction ~= 5) & (Y_test == 5));

precision_5 = tp_5 / (tp_5 + fp_5);
recall_5 = tp_5 / (tp_5 + fn_5);
score_5 = (2 * precision_5 * recall_5) / (precision_5 + recall_5);

% Compute f1 score for 6 class
tp_6 = sum((prediction == 6) & (Y_test == 6));
fp_6 = sum((prediction == 6) & (Y_test ~= 6));
fn_6 = sum((prediction ~= 6) & (Y_test == 6));

precision_6 = tp_6 / (tp_6 + fp_6);
recall_6 = tp_6 / (tp_6 + fn_6);
score_6 = (2 * precision_6 * recall_6) / (precision_6 + recall_6);

% Compute weighted score
score = 0.05*score_0 + 0.2*score_1 + 0.2*score_2 + 0.15*score_3 + 0.15*score_4 + 0.15*score_5 + 0.1*score_6;
end