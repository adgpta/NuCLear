function [selectedOption] = addCldef()

% Create UIFigure and hide until all components are created
UIFigure = uifigure('Visible', 'off');
UIFigure.Position = [1000 800 450 200];
UIFigure.Name = 'Add new class deifnitions';

% Create TextArea
TextArea = uitextarea(UIFigure);
TextArea.Placeholder = ['Neuron 0 Maj'];
TextArea.Position =  [45 58 350 76];;


% Create EnternewclassdeifnitionsseparatedbyasemicolonLabel
Enternewcl = uilabel(UIFigure);
Enternewcl.FontWeight = 'bold';
Enternewcl.FontSize = 12;
Enternewcl.Position = [45 141 533 59];
        
Enternewcl.Text = {'Enter new class deifnitions. Create a new line for each definition' ...
    'Add "Maj" for major category or "Sub" for sub category of cells. ' ...
    'Add a new line for each definition.'};

% Create CancelButton
CancelButton = uibutton(UIFigure, 'push');
CancelButton.ButtonPushedFcn =  @(btn, event) CancelButtonPushed(btn, event);
CancelButton.Position = [300 10 100 23];
CancelButton.Text = 'Cancel';

% Create NextButton
NextButton = uibutton(UIFigure, 'push');
NextButton.ButtonPushedFcn = @(btn, event) NextButtonPushed(btn, event, TextArea);
NextButton.Position = [180 10 100 23];
NextButton.Text = 'Next';

% Show the figure after all components are created
UIFigure.Visible = 'on';

selectedOption = '';

% Function executed when CancelButton is pressed
    function CancelButtonPushed(~, ~)
        disp('Dialog canceled.');
        close(UIFigure);
    end

% Function executed when NextButton is pressed
    function NextButtonPushed(~, ~, dropDown)
        selectedOption = string(dropDown.Value);
        % Save the selected option to a variable or perform any desired action
        uiresume(UIFigure); % Resume execution after the dialog box is closed
        close(UIFigure);
    end


% Wait for figure
uiwait(UIFigure);
end

