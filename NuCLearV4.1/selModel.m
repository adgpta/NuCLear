function [selectedOption] = selModel(ModelInfo)

            options = fieldnames(ModelInfo);
            items = [options(string(options) ~= 'ClassDef')];
            
            % Create UIFigure and hide until all components are created
            UIFigure = uifigure('Visible', 'off');
            UIFigure.Position = [1000 800 245 99];
            UIFigure.Name = 'Select Model';

            % Create ModelTypeDropDownLabel
            ModelTypeDropDownLabel = uilabel(UIFigure);
            ModelTypeDropDownLabel.HorizontalAlignment = 'center';
            ModelTypeDropDownLabel.Position = [24 64 67 22];
            ModelTypeDropDownLabel.Text = 'Model Type';

            % Create ModelTypeDropDown
            ModelTypeDropDown = uidropdown(UIFigure);
            ModelTypeDropDown.Items = items;
            ModelTypeDropDown.Position = [106 64 100 22];
            ModelTypeDropDown.Value = 'Orig';

            % Create CancelButton
            CancelButton = uibutton(UIFigure, 'push');
            CancelButton.ButtonPushedFcn =  @(btn, event) CancelButtonPushed(btn, event);
            CancelButton.Position = [135 10 100 23];
            CancelButton.Text = 'Cancel';

            % Create NextButton
            NextButton = uibutton(UIFigure, 'push');
            NextButton.ButtonPushedFcn = @(btn, event) NextButtonPushed(btn, event, ModelTypeDropDown);
            NextButton.Position = [10 10 100 23];
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
        selectedOption = dropDown.Value;
        % Save the selected option to a variable or perform any desired action
        uiresume(UIFigure); % Resume execution after the dialog box is closed
        close(UIFigure);
    end


    % Wait for figure
    uiwait(UIFigure);
end



