%code adapted from existing preproc pipeline from Dr. Danielle Sliva 
%clear all;
%eeglab

%eeglab specific format 
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

rawdatadir = '/Users/elizabethkaplan/Desktop/Greene/source_data';
subject_dir = '/Users/elizabethkaplan/Desktop/Greene/subject_folders/TReC_12237';
analysis_dir = '/Users/elizabethkaplan/Desktop/Greene/analysis_files';

%% 1. Load in data and add channel info 

setfile = ['TReC_12237_PreEEG.set'];   

% load the dataset into EEGLAB format 
EEG = pop_loadset('filename', setfile, 'filepath', subject_dir);

%save in this format for later visual inspection
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 1);

chanloc_file = fullfile(fileparts(which('eeglab.m')), 'sample_locs', 'standard_waveguard64.elc');
EEG = pop_chanedit(EEG, 'load', {chanloc_file, 'filetype', 'autodetect'});

%% 2. Read in timestamps and add them to each .set file as EEG markers (from Sarah's code)
% can skip this step if you are working with a .set file with existing
% timestamps (named _ts)

% Read the GP_ANT_video_offset file
%offset_file = '~/Desktop/research/projects/353_TReC/EEG/GP_ANT_video_offset.txt';
%offset_table = readtable(offset_file);

% Extract subject ID and timepoint (PreEEG/PostEEG) from the filename using regular expression
%filename = D(iFile).name; 
subjectID = 'TReC_12237'; % Extract subject ID (e.g. TReC_12237)
timepoint = 'PreEEG'; % Extract timepoint (PreEEG or PostEEG)
    
if isempty(subjectID) || isempty(timepoint)
  error('Could not extract subjectID or timepoint from filename: %s', filename);
end
    
% Find the corresponding GP_ANT_video_offset for this subject and timepoint
%offset_row = offset_table(strcmp(offset_table.SubjectID, subjectID) & strcmp(offset_table.Time, timepoint), :);
    
%if isempty(offset_row)
%    % If no offset data is found, skip this subject and timepoint
%    fprintf('No GP_ANT_video_offset found for subject %s, timepoint %s. Skipping...\n', subjectID, timepoint);
%end
    
%GP_ANT_video_offset = offset_row.Offset;  % Set the GP_ANT_video_offset

% Load the corresponding timestamp file 
tic_event_file = '/Users/elizabethkaplan/Desktop/Greene/subject_folders/TReC_12237/TReC_12237_PreEEG_all_timestamps_sorted2.txt';
    
if ~isfile(tic_event_file)
    error('Timestamp file not found for subject %s, timepoint %s', subjectID, timepoint);
end

tic_event_table = readtable(tic_event_file);

% Check for video event marker
video_code_idx = [];
for iEv = 1:numel(EEG.event)
    if contains(EEG.event(iEv).type, 'Video') % Check if event type contains 'Video'
        video_code_idx = [video_code_idx; iEv];
    end
end
    
if numel(video_code_idx) > 1 
    error('More than 1 video code found, I give up!');
end

% Get the video event time in milliseconds
video_code_time = EEG.event(video_code_idx).latency / EEG.srate;

% Calculate the tic event time
% deleted GP_ANT latency as Tabitha already accounted for it
tic_event_time_1 = (tic_event_table.latency) * 60; % convert Tabitha's timestamps to seconds
tic_event_time = tic_event_time_1 + video_code_time; % add video code time to timestamps

% Insert events into EEG dataset
for iEv = 1:numel(tic_event_time)
   eventType = tic_event_table{iEv, 2};
   EEG = pop_editeventvals(EEG, 'insert', {1, [], [], []}, ...
      'changefield', {1, 'latency', tic_event_time(iEv)}, ...
      'changefield', {1, 'type', eventType}, ...
      'changefield', {1, 'duration', 0});
   EEG = eeg_checkset(EEG);
end

%% 3. Read in non-overlapping tics across both conditions and create single tic_event table
% add in premonitory event markers 

ticfree_file = '/Users/elizabethkaplan/Desktop/Greene/subject_folders/TReC_12237/TReC_12237_PreEEG_TicFree_ticsInclude.txt';
suppress_file = '/Users/elizabethkaplan/Desktop/Greene/subject_folders/TReC_12237/TReC_12237_PreEEG_TicSuppress_ticsInclude.txt';

ticfree = readtable(ticfree_file, 'Delimiter', '\t');  
suppress = readtable(suppress_file, 'Delimiter', '\t');

% Combine into one table & sort by onset
tic_events = [ticfree; suppress];
tic_events = sortrows(tic_events, 'onset');

%sampling rate
fs = EEG.srate;

% Adjust onset and offset times by adding video offsets, convert to seconds
adjusted_onset_sec = (tic_events.onset + video_code_time) / 1000;
adjusted_offset_sec = (tic_events.offset + video_code_time) / 1000;

% Add adjusted times to tic_events table
tic_events.onset_sec = adjusted_onset_sec;
tic_events.offset_sec = adjusted_offset_sec;

%add in pre_tic event markers for non-overlapping tics 
%for iEv = 1:height(tic_events)
 %   preTicTime = tic_events.onset_sec(iEv) - 2.5;

    % Only add if it doesn't go before EEG start
  %  if preTicTime > 0
   %     EEG = pop_editeventvals(EEG, ...
    %        'insert', {1, [], [], []}, ...
     %       'changefield', {1, 'latency', preTicTime}, ...
      %      'changefield', {1, 'type', 'PreTic'}, ...
       %     'changefield', {1, 'duration', 0});
   % end

    %EEG = eeg_checkset(EEG);
%end

%% 3. Remove DC offset (channel mean) 

for iChan = 1:size(EEG.data, 1)
    EEG.data(iChan, :) = single(double(EEG.data(iChan, :)) - ...
                                mean(double(EEG.data(iChan, :))));
end

%% 4. Notch filter 60 Hz - to remove line noise (noise from electricity at 60 Hz)

EEG_params.notch.low_cutoff = 58; 
EEG_params.notch.high_cutoff = 62;
notch_filter = 1;

EEG = pop_eegfiltnew(EEG, EEG_params.notch.low_cutoff, ...
    EEG_params.notch.high_cutoff, [], notch_filter);
EEG = eeg_checkset(EEG);

% Save updated EEG_params
%save(params_file, 'EEG_params');

%% 5. High and low pass filtering - adjust as needed

EEG = pop_basicfilter(EEG, 1:EEG.nbchan, ...
        'Cutoff' , .1, ...
        'Design' , 'butter' , ...
        'Filter' , 'highpass', ...
        'Order'  , 2 , ...
        'RemoveDC','on', ...
        'Boundary','none');

EEG = pop_basicfilter(EEG, 1:EEG.nbchan, ...
        'Cutoff' , 70 , ...
        'Design'  , 'butter' , ...
        'Filter'  , 'lowpass', ...
        'Order'   , 4 , ...
        'Boundary', 'none');

%% 7. Remove premonitory urge and tic

%sampling rate
fs = EEG.srate;

premonitory_sec = 2.5; % 2.5 seconds before tic 
premonitory_samples = round(premonitory_sec * fs);

tic_segments = {};
tic_ranges = [];
premonitory_ranges = [];

% calculate range of data to remove from EEG structure
for i = 1:height(tic_events)
    % Convert adjusted onset/offset times to samples
    onset_sample = round(tic_events.onset_sec(i) * fs);
    offset_sample = round(tic_events.offset_sec(i) * fs);

    % Account for premonitory urge period
    start_sample = onset_sample - premonitory_samples;
    end_sample = offset_sample;

    tic_ranges = [tic_ranges; onset_sample, end_sample];
    premonitory_ranges = [premonitory_ranges; start_sample, onset_sample];
end  

% convert sample ranges to time in seconds
tic_times_to_remove = tic_ranges / fs;
premonitory_range_to_keep = premonitory_ranges / fs;

% Extract premonitory segments into a new EEG structure
EEG_premonitory = pop_select(EEG, 'time', premonitory_range_to_keep);
EEG_premonitory.setname = 'EEG_premonitory_segments';

% Remove tic segments 
EEG_clean = pop_select(EEG, 'notime', tic_times_to_remove);

% Remove events outside new data bounds
max_latency = EEG_clean.pnts;
EEG_clean.event = EEG_clean.event([EEG_clean.event.latency] <= max_latency);

% Store updated EEG datasets
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG_clean, CURRENTSET);

%% 8. Run sliding window  

fs = EEG.srate;               
epochLength_sec = 3;   % Epoch length in seconds
epochLength_samples = epochLength_sec * fs;

nSamples = size(EEG.data, 2); 
nEpochs = floor(nSamples / epochLength_samples);

% Start indexing after existing events - appends new events rather than
% rewritting existing (tic time stamps). Bin the data around dummy events, based on epoch length defined above  
nExistingEvents = length(EEG.event);

for i = 1:nEpochs
    idx = nExistingEvents + i;
    EEG.event(idx).latency = (i-1) * epochLength_samples + 1;  
    EEG.event(idx).type = 'dummy';                             
end

EEG = eeg_checkset(EEG, 'eventconsistency');

% Find all boundary events and rename them temporarily. if boundary is stored
% as 'boundary' in data, sliding window will auto reject any bin that
% contains a boundary
boundary_idx = find(strcmp({EEG.event.type}, 'boundary'));
for i = 1:length(boundary_idx)
    EEG.event(boundary_idx(i)).type = 'noboundary';
end

% Create dummy EVENTLIST to satisfy ERPLAB functions
EEG = pop_creabasiceventlist(EEG, ...
    'Eventlist', '', ...
    'BoundaryNumeric', {-99}, ...
    'BoundaryString', {'boundary'});

%actually epoch the data
EEG = pop_epoch(EEG, {'dummy'}, [0 epochLength_sec], 'epochinfo', 'yes');
EEG = eeg_checkset(EEG);

fprintf('Number of epochs after epoching: %d\n', EEG.trials);

%sliding window params 
EEG_params.reject.Threshold = 225;      
EEG_params.reject.Twindow = [0 epochLength_sec * 1000];   % this line will auto update with epoch length 
EEG_params.reject.Windowsize = 50;      
EEG_params.reject.Windowstep = 25;      

% Run sliding window
EEG = pop_artmwppth(EEG, ...
    'Channel', 1:size(EEG.data,1), ...
    'Flag', [1 3], ...
    'Threshold', EEG_params.reject.Threshold, ...
    'Twindow', EEG_params.reject.Twindow, ...
    'Windowsize', EEG_params.reject.Windowsize, ...
    'Windowstep', EEG_params.reject.Windowstep);

% Check how many epochs were rejected
fprintf('Rejected %d out of %d epochs (%.2f%%)\n', ...
    sum(EEG.reject.rejmanual), EEG.trials, ...
    100 * sum(EEG.reject.rejmanual) / EEG.trials);

% visualize rejected portions -- these will be highlighted in yellow 
pop_eegplot(EEG, 1, 1, 1);

%% Return data to continuous structure 

epochLength_samples = EEG.pnts;  % Number of samples per epoch
bad_epochs = find(EEG.reject.rejmanual);  % Get indices of rejected epochs

% Compute start and end sample ranges for each bad epoch
rejected_sample_ranges = [];
for i = 1:length(bad_epochs)
    start_sample = (bad_epochs(i)-1) * epochLength_samples + 1;
    end_sample = start_sample + epochLength_samples - 1;
    rejected_sample_ranges = [rejected_sample_ranges; start_sample, end_sample];
end

%use earlier saved 'EEG_clean' in order to return data to continuous, but
%apply rejection flags to the EEG structure and remove these timepoints 
for i = size(rejected_sample_ranges,1):-1:1
    EEG_clean = pop_select(EEG_clean, 'notime', ...
        [rejected_sample_ranges(i,1)/EEG.srate, rejected_sample_ranges(i,2)/EEG.srate]);
end

%update EEG Structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG_clean, CURRENTSET);


%% reject artifacts within epoched data -- ERPlab -- exact code from MP 
% Do not run for our pipeline!! 

%EEG_params.reject.Threshold = 100;
%EEG_params.reject.Twindow = EEG_params.epoch_timewindow;
%EEG_params.reject.Windowsize = 200;
%EEG_params.reject.Windowstep = 100;

%EEG  = pop_artmwppth( EEG , 'Channel',  1:31, 'Flag', [1 2], ...
   % 'Threshold',  EEG_params.reject.Threshold, ...
   % 'Twindow', EEG_params.reject.Twindow, ...
   % 'Windowsize',  EEG_params.reject.Windowsize, ...
   % 'Windowstep',  EEG_params.reject.Windowstep ); 
%EEG = eeg_checkset( EEG );
%save(params_file,'EEG_params');

%% 9. Remove bad channels 
%specifically looking for channels that do not follow pattern of power
%spectra plot
figure; pop_spectopo(EEG, 1, [], 'EEG' , 'percent',15,'freq',...
        [10 20 30],'freqrange',[1 70],'electrodes','off');

% Signal

pop_eegplot( EEG, 1, 1, 1);

disp('----------------------')
disp('----------------------')
disp('Manual Input Required')
disp('----------------------')
disp('----------------------')
bad = input('Select bad channels:');

EEG=pop_select(EEG, 'nochannel',bad);

%% 10. ICA to remove blinks and muscle artifact 

%create copy of the data highpassed at 1 Hz - improves identification of
%eye blinks and extreme muscle artifacts (aka smooths out signal so these are
%more obvious)
EEG = pop_basicfilter(EEG,1:EEG.nbchan,'Cutoff',1,'Filter','highpass','Design','butter','Order',2);

% Down‑sample to 256 Hz first (faster)
EEG = pop_resample(EEG,256);

%last input is the num of ICs you want computed -- this number must be less
%than the number of channels you actually have so may need updating
%depending on how many channels removed 
EEG = pop_runica(EEG,'icatype','picard','pca',60);

%longer ICA version - takes alot more time to run
%EEG_ica = pop_runica(EEG_ica,'extended',1,'icatype','runica','pca',EEG_ica.nbchan);

%convert weights back to original dataframe
%EEG.icaweights = EEG.icaweights;
%EEG.icasphere  = EEG_ica.icasphere;
%EEG.icachansind  = EEG_ica.icachansind;
%EEG.icawinv = EEG_ica.icawinv;
%EEG = eeg_checkset(EEG);

%% 11. Visualize ICs - label with ICLabel and inspect each IC for accuracy - and reject 

figure;
pop_selectcomps(EEG, 1:35);  % Show the first 35 components (you can adjust this range if needed, but components larger than this are more unreliable)
pop_eegplot(EEG, 0, 1, 1);    

disp('----------------------')
disp('----------------------')
disp('Manual Input Required: Select Components to Reject')
disp('----------------------')
disp('----------------------')

% Prompt for user input to select components to reject
% You can select multiple components using a vector (e.g., [1 2 5 8])
reject_components = input('Enter the list of components to reject (e.g., [1 2 5 8]): ');

% Print the list of rejected components to the output for reference
%txt = sprintf('Reject Components: %s\n', num2str(reject_components));
%fprintf(out_txt, '%s\n ', txt); 
%fprintf('%s', txt);

% Remove the selected components
EEG = pop_subcomp(EEG, reject_components, 0); 

%% 12. re-reference to CPz 
ref_chans = eeg_decodechan(EEG.chanlocs,{'CPz'});
EEG = pop_reref(EEG,ref_chans,'keepref','off');

%% 13. Interpolate the rejected components using spherical interpolation - havent ran this part on any yet!
%EEG = pop_interp(EEG, channels, 'spherical'); 

% Re-reference the data, excluding certain channels (e.g., 64 and 65)
%EEG = pop_reref(EEG, [], 'exclude', [64 65]); 

% Change directory back to the initial directory (if needed)
%cd(init_dir);

%% 14. Visual Inspection of data - "final pass" 
%can manually remove bad sections of data - be sure to hit "reject" in the bottom right corner for EEGLAB to
%automatically remove marked segments

%downsample for plotting - be sure to resample back to 500 Hz
%EEG = pop_resample(EEG,250);

EEG = eeg_checkset(EEG);

pop_eegplot( EEG, 1, 1, 1);
pop_eegplot( EEG, 1, 1, 1);
eegh

%% Save data with chanloc info to local device

EEG = pop_saveset(EEG, 'filename', 'TRec_masked', 'filepath', '/Users/elizabethkaplan/Desktop/Greene/');

%% Visual inspection 

eegplot(EEG.data, 'srate', EEG.srate, 'eloc_file', EEG.chanlocs, 'dispchans', EEG.nbchan, 'events', EEG.event);

%eegplot(EEG_old.data, 'srate', EEG_old.srate, 'eloc_file', EEG_old.chanlocs, 'dispchans', EEG_old.nbchan, 'events', EEG_old.event);
