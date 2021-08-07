function data = analyze_neurint(filenames)
%%
if isempty(mglGetSID)
  disp(sprintf('No SID set. Set subject ID using mglSetSID to continue.'));
  keyboard
end

datadir = sprintf('~/data/neurint/%s', mglGetSID);

% get the files list
if ieNotDefined('filenames')
  files = dir(fullfile(sprintf('%s/2*stim*.mat',datadir)));
  filenames = {files.name};
elseif ischar(filenames)
  filenames = {filenames};
elseif ~iscell(filenames)
  disp('Unrecognized format for filenames');
  return;
end

if length(filenames) == 0
    disp(sprintf('No data found'));
    return;
end

count = 1; 
data = struct('response', [], 'reaction_time', [], ...
              'nTrials', 0, 'nValTrials', 0, 'accByRuns', []);
for fi = 1:length(filenames)
  disp(filenames{fi});
  load(fullfile(sprintf('~/data/neurint/%s/%s',mglGetSID,filenames{fi})));
  
  e = getTaskParameters(myscreen,task);
  if iscell(e), e = e{1}; end
  
  if e.nTrials>1
    f = fields(e.parameter);
    for j = 1:length(f)
        if ~isfield(data, f{j})
            data.(f{j}) = [];
        end
        data.(f{j}) = [data.(f{j}) e.parameter.(f{j})];
    end
    f = fields(e.randVars);
    for j = 1:length(f)
        if ~isfield(data, f{j})
            data.(f{j}) = [];
        end
        data.(f{j}) = [data.(f{j}) e.randVars.(f{j})];
    end
    
    data.reaction_time = [data.reaction_time e.reactionTime];
    data.nTrials = data.nTrials + e.nTrials;

    % Calculate number of valid trials by excluding eye movement trials and no-response trials.
    data.nValTrials = data.nValTrials + sum(~isnan(e.response));
    data.accByRuns = [data.accByRuns nanmean(e.randVars.correct)];
    
  end
  count = count + 1;
end
disp(sprintf('SUBJECT %s: Found %i runs with a total of %i trials', mglGetSID, length(data.accByRuns), data.nTrials));

%% Combine across left and right side and code one image as the "left" choice and the other as the "right"
rightwardChoice = data.imgChoice-1;
intervals = [0,25,50,75,100];
mean_response = zeros(length(intervals),length(stimulus.feature_spaces));
CI_response = zeros(length(intervals),length(stimulus.feature_spaces));
for i = 1:length(stimulus.feature_spaces)
    for j = 1:length(intervals)
        intervalRightChoices = rightwardChoice(data.centerInterval==intervals(j) & data.feature_space==i);
        mean_response(j,i) = nanmean(intervalRightChoices);
        CI_response(j,i) = 1.96*nanstd(intervalRightChoices) / sqrt(sum(~isnan(intervalRightChoices)));
    end
end

%%
figure;
colors = brewermap(7, 'Spectral');
handles = [];
for i = 1:size(mean_response,2)
    %subplot(2,size(mean_response,2)/2,i);
    h = myerrorbar(intervals, mean_response(:,i), 'yError', CI_response(:,i), 'Symbol', 'o', 'Color', colors(i,:)); 
    handles = [handles h];
    hold on;
    % Plot fit line
    c = polyfit(intervals, mean_response(:,i)', 1);
    X = linspace(intervals(1), intervals(end), 100);
    y_est = polyval(c, X);
    plot(X, y_est, '-', 'Color', colors(i,:));
    %hline(0.5, ':k');
end
legend(handles, strrep(stimulus.feature_spaces, '_', '-->'));
%title(strrep(stimulus.feature_spaces{i}, '_', '-->'));
set(gca, 'XTick', intervals);
xlabel('Percentage of interpolation in representational space');
ylabel('Proportion of rightward choices');
ylim([0,1]);
xlim([-10, 110]);
drawPublishAxis('labelFontSize', 14);

%%
keyboard

%% Split performance by different image pairs

rightwardChoice = data.imgChoice-1;
intervals = [0,25,50,75,100];
image_pairs = unique(data.image_pair);
mean_response = zeros(length(intervals),length(stimulus.feature_spaces), length(image_pairs));
CI_response = zeros(length(intervals),length(stimulus.feature_spaces), length(image_pairs));
for i = 1:length(stimulus.feature_spaces)
    for j = 1:length(intervals)
        for k = image_pairs
            intervalRightChoices = rightwardChoice(data.centerInterval==intervals(j) & data.feature_space==i & data.image_pair==k);
            mean_response(j,i,k) = nanmean(intervalRightChoices);
            CI_response(j,i,k) = 1.96*nanstd(intervalRightChoices) / sqrt(sum(~isnan(intervalRightChoices)));
        end
    end
end

figure;
colors = brewermap(7, 'Spectral');
handles = [];
for k = image_pairs
    subplot(length(image_pairs),1,k);
    for i = 1:size(mean_response,2)
        %subplot(2,size(mean_response,2)/2,i);
        h = myerrorbar(intervals, mean_response(:,i,k), 'yError', CI_response(:,i,k), 'Symbol', 'o', 'Color', colors(i,:)); 
        handles = [handles h];
        hold on;
        % Plot fit line
        c = polyfit(intervals, mean_response(:,i,k)', 1);
        X = linspace(intervals(1), intervals(end), 100);
        y_est = polyval(c, X);
        plot(X, y_est, '-', 'Color', colors(i,:));
        
        %hline(0.5, ':k');
    end
    title(strrep(stimulus.image_pairs{k}, '_', '-'));
    legend(handles, strrep(stimulus.feature_spaces, '_', '-->'));
    %title(strrep(stimulus.feature_spaces{i}, '_', '-->'));
    set(gca, 'XTick', intervals);
    xlabel('Percentage of interpolation in representational space');
    ylabel('Proportion of rightward choices');
    ylim([0,1]);
    xlim([-10, 110]);
    drawPublishAxis('labelFontSize', 14);

end

%% Bin data and plot performance
numBins = 10;
rightwardChoice = data.response-1;
mean_response = zeros(numBins, length(intervals));
CI_response = zeros(numBins, length(intervals));
bins = [1:(length(data.response)/numBins):length(data.response) length(data.response)];
for i = 1:numBins
    rightwardChoice = data.response(bins(i):bins(i+1))-1;
    for j = 1:length(intervals)
        intervalRightChoices = rightwardChoice(data.centerInterval(bins(i):bins(i+1))==intervals(j));
        mean_response(i,j) = nanmean(intervalRightChoices);
        CI_response(i,j) = 1.96*nanstd(intervalRightChoices) / sqrt(sum(~isnan(intervalRightChoices)));
    end
end

colors = brewermap(numBins,'Blues');
figure;
for i=1:numBins
    myerrorbar(intervals, mean_response(i,:), 'yError', CI_response, 'Symbol', 'o', 'Color', colors(i,:)); hold on;
    c = polyfit(intervals, mean_response(i,:), 1);
    X = linspace(intervals(1), intervals(end), 100);
    y_est = polyval(c, X);
    plot(X, y_est, '-', 'color', colors(i,:));
end

function out = sigmoidResid(params, X, R)

predicted = sigmoid(X, params);
out = sum((R - predicted).^2);

function out = sigmoid(X, params)
out =  params(3) + (params(2) ./ (1 + exp(-params(1)*(X-params(4)))));
