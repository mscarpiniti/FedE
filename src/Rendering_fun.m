% Rendering_fun is a customizable library function that 
% performs the graphic rendering of the results of
% an already executed FedE run. For this purpose, the script loads the
% file with the workspace of an already run FedE script and, then, it executes
% the rendering routine on the loaded data. Rendering_fun allows the 
% data rendering of an already executed FedE script without to re-run 
% it from scratch. 


clear; close all; fprintf('\n'); clc;
%
filestring= 'FedE_trained_MLP1_workspace.mat';
%String of the file storing the workspace to be loaded
load(filestring);
%Load the workspace of an already executed FedE run
fprintf('Message from Rendering_fun: the rendering of the data stored in the workspace file named as: ''%s'' is in execution\n', filestring);
fprintf('Message from Rendering_fun: the running time of the FedE run whose workspace is stored in the file: ''%s'' was been of %g (sec)\n', filestring, round(FedE_running_time,2));
%
%---------------------------End loading section----------------------------
%
%---------------------------Rendering section------------------------------
%
fignumber = 0;
%
fignumber = fignumber+1;
figure(fignumber);
%Plot the behavior of the test accuracy obtained during the training phase versus global clock sequence
if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
end %End if 

augmented_accuracy_vec = gl_trainedweight_vec; 

if ((gl_clock_vec_aux(end)) > 0)
        aug_x_vec = 0.0: 0.05: gl_clock_vec_aux(end); %Augmented global clock vector
        int_augmented_accuracy_vec= spline(gl_clock_vec_aux, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end %End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Behavior of global test accuracy during the training phase-vs.-wall clock time'); 
hold on;
[best_acc, best_acc_round] = max(int_augmented_accuracy_vec);
fprintf('The best accuracy attained by the global model is %.2f (%%) at wall-clock time %.1f (sec)\n', best_acc, aug_x_vec(best_acc_round(1)));
plot(aug_x_vec(best_acc_round(1)), best_acc, 'k*');
legend({sprintf('Test accuracy of the global model\n'),sprintf('Best accuracy\n')},'Location','SouthEast');
hold off
%
fignumber = fignumber+1;
figure(fignumber);
%Plot the behaviors of global objective function, squared norms of global 
%primal variables and, if present, squared norms of global dual variables
%versus global clock sequence

if (dual_vec_length >=1 ) %Dual variables are used
    %
    if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
    end %End if 
    %
    if ((gl_clock_vec_aux(end)) > 0)
        aug_x_vec = 0.0: 0.05: gl_clock_vec_aux(end); %Augmented global clock vector
        int_augmented_or_obfun_vec = spline(gl_clock_vec_aux, augmented_or_obfun_vec_aux,aug_x_vec);
        int_augmented_or_squared_primal_variables_vec = spline(gl_clock_vec_aux, augmented_or_squared_primal_variables_vec_aux,aug_x_vec);
        int_augmented_or_squared_dual_variables_vec = spline(gl_clock_vec_aux, augmented_or_squared_dual_variables_vec_aux,aug_x_vec);
    else
      aug_x_vec = 0.0;
      int_augmented_or_obfun_vec = augmented_or_obfun_vec_aux;
      int_augmented_or_squared_primal_variables_vec = augmented_cp_or_squared_primal_variables_vec_aux;
      int_augmented_or_squared_dual_variables_vec = augmented_cp_or_squared_dual_variables_vec_aux;
    end %End if
    %
    subplot(4,1,1)
    plot(aug_x_vec, max(round(or_gl_min_ob_function_aux,3), min(int_augmented_or_obfun_vec_aux,round(or_gl_max_ob_function_aux,3))), 'r-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Training loss-vs.-wall clock time'); 
    legend({sprintf('Training loss\n')},'Location','NorthEast');
    %
    subplot(4,1,2)
    plot(aug_x_vec,max(0,(1/primal_vec_length)*min(int_augmented_or_squared_primal_variables_vec_aux,int_augmented_or_squared_primal_variables_vec_aux(round((7/8)*length(int_augmented_or_squared_primal_variables_vec_aux))))),'g-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Behavior of the per-entry average squared norm of the global primal model-vs.-wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global primal model\n')},'Location','NorthEast');
    %
    subplot(4,1,3)
    plot(aug_x_vec,max(0,(1/dual_vec_length)*int_augmented_or_squared_dual_variables_vec_aux),'b-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Behavior of the per-entry average squared norm of the global dual model-vs.-wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global dual model\n')},'Location','NorthEast');
    %
   subplot(4,1,4)
   %Plot the behavior of the test accuracy obtained during the training phase versus global clock sequence
if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
end%End if 

augmented_accuracy_vec = gl_trainedweight_vec; 

if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
end %End if 

if ((gl_clock_vec_aux(end)) > 0)
        aug_x_vec = 0.0: 0.05: gl_clock_vec_aux(end); %Augmented global clock vector
        int_augmented_accuracy_vec= spline(gl_clock_vec_aux, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end %End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Global test accuracy during the training phase -vs.-wall clock time');
hold on
plot(aug_x_vec(best_acc_round(1)), best_acc, 'k*');
legend({sprintf('Test accuracy of the global model\n'), sprintf('Best accuracy\n')},'Location','SouthEast');
hold off
%
end %End if
%
if (dual_vec_length == 0 ) %Dual variables are NOT used
    %
    if(or_gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior;
    end %End if 
    %
    if (gl_clock_vec_aux(end) > 0)
        aug_x_vec = 0.0: 0.05: gl_clock_vec_aux(end); %Augmented global clock vector
        int_augmented_or_obfun_vec_aux= spline(gl_clock_vec_aux, augmented_or_obfun_vec_aux,aug_x_vec);
        int_augmented_cp_or_squared_primal_variables_vec_aux= spline(gl_clock_vec_aux, augmented_or_squared_primal_variables_vec_aux,aug_x_vec);
    else
        aug_x_vec = 0.0;
        int_augmented_or_obfun_vec_aux = augmented_or_obfun_vec_aux;
        int_augmented_or_squared_primal_variables_vec_aux = augmented_or_squared_primal_variables_vec_aux;
    end %End if
    %
    subplot(3,1,1)
    plot(aug_x_vec,int_augmented_or_obfun_vec_aux,'r-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Behavior of the global training loss function-vs.-continuos-valued wall clock time'); 
    legend({sprintf('Global training loss function\n')},'Location','NorthEast');
    %
    subplot(3,1,2)
    plot(aug_x_vec,max(0,(1/primal_vec_length)*min(int_augmented_or_squared_primal_variables_vec_aux,int_augmented_or_squared_primal_variables_vec_aux(round((7/8)*length(int_augmented_or_squared_primal_variables_vec_aux))))),'g-');
    xlabel('Wall clock time (sec)'); ylabel('');
    grid on;
    title('FedE - Behavior of the per-entry average squared norm of the global primal variables -vs.- continuos-valued wall clock time'); 
    legend({sprintf('Per-entry average squared norm of the global primal variables\n')},'Location','northeast');
    %
    subplot(3,1,3)
    %Plot the behavior of the test accuracy obtained during the training phase versus global clock sequence
if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
end %End if 

augmented_accuracy_vec = gl_trainedweight_vec; 

if(gl_clock_vec_aux(end) == Inf)
       aux_vec = gl_clock_vec_aux;
       aux_vec(end) = aux_vec(end-1) + 0.1;
       gl_clock_vec_aux = aux_vec; %Avoid ill behavior
end %End if 

if ((gl_clock_vec_aux(end)) > 0)
        aug_x_vec = 0.0: 0.05: gl_clock_vec_aux(end); %Augmented global clock vector
        int_augmented_accuracy_vec = spline(gl_clock_vec_aux, augmented_accuracy_vec, aug_x_vec);
else
      aug_x_vec = 0.0;
      int_augmented_accuracy_vec = augmented_accuracy_vec;
end %End if
%
plot(aug_x_vec, min(100,max(0,int_augmented_accuracy_vec)), 'r-');
xlabel('Wall clock time (sec)'); ylabel('Accuracy (%)');
grid on;
title('FedE - Behavior of global test accuracy during the training phase  -vs.- continuos-valued wall clock time'); 
legend({sprintf('Global Test Accuracy\n')},'Location','SouthEast');
%
end %End if
%
%Bar plots of the best local accuracy obtained by the sequence of the local
%models trained by each coworker
fignumber = fignumber+1;
figure(fignumber);
b = bar([1:1:coworker_number],local_best_accuracy_vec, 0.02,'grouped','FaceColor','flat');
for j=1:length(local_best_accuracy_vec)
    b.CData(j,:)= [(j/(length(local_best_accuracy_vec))), 0.01*(j/(length(local_best_accuracy_vec))), 0.01*(j/(length(local_best_accuracy_vec)))];
    %Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Accuracy (%)');
title('FedE - Best test accuracy attained by the sequence of the local models trained by each coworker'); 
%
%Bar plots of the final local accuracy obtained by the final local
%models trained by each coworker
fignumber = fignumber+1;
figure(fignumber);
b = bar([1:1:coworker_number],local_final_accuracy_vec, 0.02,'grouped','FaceColor','flat');
for j=1:length(local_final_accuracy_vec)
    b.CData(j,:)= [(j/(length(local_final_accuracy_vec))), 0.01*(j/(length(local_final_accuracy_vec))), 0.01*(j/(length(local_final_accuracy_vec)))];
    %Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Accuracy (%)');
title('FedE - Test accuracy attained by the final local model trained by each coworker'); 
%
%
%Plot the behaviors of the squared norms of local 
%primal variables and, if present, squared norms of local dual variables
%versus the index of the performed local primal/dual iterations
if(size(cw_squared_norm_primal_variables_mtx_aux,2) > 1)
            aug_x_vec = 1.0: 0.05: (size(cw_squared_norm_primal_variables_mtx_aux,2)); %Augmented global clock vector
 else
            aug_x_vec = 1.0;
end %End if 

for j = 1: floor(coworker_number/2) : coworker_number
    fignumber = fignumber+1;
    figure(fignumber);
    if (dual_vec_length >=1 ) %Local dual variables are used
        subplot(3,1,1)
        aux_y_vec = linspace(0, training_wall_clock_time, length_cp_cw_local_accuracy_vec(j));
        int_x1_aux = 0.0: 0.05: training_wall_clock_time; %Augmented global clock vector
        int_y_aux_vec = max( 0 , min(100, spline(aux_y_vec,local_accuracy_mtx(j, 1:length_cp_cw_local_accuracy_vec(j)), int_x1_aux)));
        plot(int_x1_aux(1:end), int_y_aux_vec(1:end), 'r-');
        xlabel('Wall clock time (sec)'); ylabel('Accuracy(%)');
        grid on;
        title(sprintf('FedE - Test accuracy of the local model trained by Coworker n. %d -vs.- global wall clock time\n', j)); 
        legend({sprintf('Accuracy of the local models trained by Coworker n. %d\n', j)},'Location','SouthEast');  
        %
        subplot(3,1,2)
        length_aux = 0;

        while(1)
             if (((length_aux + 1) <= length(cw_squared_norm_primal_variables_mtx_aux(j, 1:end))) && (cw_squared_norm_primal_variables_mtx_aux(j, (length_aux + 1)) > 0))
                length_aux = length_aux +1;
             else
                 break; %Exit while loop
             end %End if   
        end %End while
        plot(1:length_aux, max(0,(1/primal_vec_length)*cw_squared_norm_primal_variables_mtx_aux(j,1:length_aux)),'g-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local primal variables -vs.- the local integer-valed iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local primal variables - Coworker n. %d\n', j)},'Location','NorthEast'); 
        %
        subplot(3,1,3)
        length_aux1 = 0;

        while(1)
             if (((length_aux1 + 1) <= length(cw_squared_norm_dual_variables_mtx_aux(j, 1:end))) && (cw_squared_norm_dual_variables_mtx_aux(j, (length_aux1 + 1)) >= 0))
                length_aux1 = length_aux1 +1;
             else
                 break; %Exit while loop
             end %End if   
        end %End while

        plot(1:min(length_aux1, length_aux), max(0,(1/dual_vec_length)*cw_squared_norm_dual_variables_mtx_aux(j,1:min(length_aux1, length_aux))),'b-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local dual variables -vs.- the local integer-valed iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local dual variables - Coworker n. %d\n',j)},'Location','NorthEast');

    end %End if
%
    if (dual_vec_length == 0 ) %Local dual variables are NOT used
        %
        if(size(cw_squared_norm_primal_variables_mtx_aux,2) > 1)
            aug_x_vec = 1.0: 0.05: (size(cw_squared_norm_primal_variables_mtx_aux,2)); %Augmented global clock vector
        else
           aug_x_vec = 1.0;
        end %End if
        %
        subplot(2,1,1)
        aux_y_vec = linspace(0, training_wall_clock_time, length_cp_cw_local_accuracy_vec(j));
        int_x1_aux = 0.0: 0.05: training_wall_clock_time; %Augmented global clock vector
        int_y_aux_vec = max( 0 , min(100, spline(aux_y_vec,local_accuracy_mtx(j, 1:length_cp_cw_local_accuracy_vec(j)), int_x1_aux)));
        plot(int_x1_aux(1:end), int_y_aux_vec(1:end), 'r-');
        xlabel('Wall clock time (sec)'); ylabel('Accuracy(%)');
        grid on;
        title(sprintf('FedE - Test accuracy of the local model trained by Coworker n. %d -vs.- global wall clock time\n', j)); 
        legend({sprintf('Accuracy of the local models trained by Coworker n. %d\n', j)},'Location','SouthEast');  
        %
        subplot(2,1,2)
        length_aux = 0;

        while(1)
             if (((length_aux + 1) <= length(cw_squared_norm_primal_variables_mtx_aux(j, 1:end))) && (cw_squared_norm_primal_variables_mtx_aux(j, (length_aux + 1)) > 0))
                length_aux = length_aux +1;
             else
                 break; %Exit while loop
             end %End if   
        end %End while

        plot(1:length_aux, max(0,(1/primal_vec_length)*cw_squared_norm_primal_variables_mtx_aux(j,1:length_aux)),'b-');
        xlabel('Local iteration index'); ylabel('');
        grid on;
        title(sprintf('FedE - Interpolated behaviors of the per-entry average squared norm of the local primal variables -vs.- local integer-valued iteration index - Coworker n. %d\n', j)); 
        legend({sprintf('Per-entry average squared norm of the local primal variables - Coworker n. %d\n', j)},'Location','NorthEast');
        %
    end %End if
%
end %End for
%
%
%Plot the local clock sequences vs. the corresponding local iteration index
%sequences
for j = coworker_number: -floor(coworker_number/2) : 1
    fignumber = fignumber +1;
    figure(fignumber);
    plot(cw_send_it_ID_mtx_aux(j,:),cw_stamptimes_mtx_aux(j,:), 'rh');
    xlabel('Integer-valued local iteration index'); 
    ylabel('Continuos-valued local clock (sec)');
    grid on;
    title(sprintf('FedE - Local clock values at which coworker n. %d succesfully sent packets to the Orchestrator -vs.- corresponding local iteration indexes - The highest clock entry is the FINAL LOCAL CLOCK TIME', j));  
end %End for
%
%
%Plot the global clock sequence vs. the corresponding global iteration index
%sequence
fignumber = fignumber +1;
figure(fignumber);
plot(0:numel(gl_clock_vec_aux)-1, gl_clock_vec_aux, 'rh');   
xlabel('Round Index');
ylabel('Wall Clock (sec)');
grid on;
title(sprintf('FedE - Wall Clock-vs.-Round Index'));
%
%Bar plots of the per-coworker numbers of transmit, sent, lost and received
%data-packets
fignumber = fignumber +1;
figure(fignumber);
%
b = bar([1:1:coworker_number],aux_mtx1,0.02,'stacked');
grid on; xlabel('Coworker index'); ylabel('Number of packets');
title('FedE - Per-coworker numbers of transmit, succesfully sent, lost and received data-packets'); 
legend({sprintf('# TX packets'),sprintf('# Succesfully sent packets'),sprintf('# RX packets'),sprintf('# Lost packets')},'Location','NorthEast'); 
%
%Bar plots of the per-coworker numbers of fictitious and ghost
%TX-packets
fignumber = fignumber +1;
figure(fignumber);
%
b = bar([1:1:coworker_number],aux_mtx2,0.02,'stacked');
xlabel('Coworker index'); ylabel('Number of packets');
grid on;
title('FedE - Per-coworker numbers of sent fictitious and ghost packets'); 
legend({sprintf('# Fictitious packets'),sprintf('# Ghost packets')},'Location','NorthEast'); 
%
%Bar plots of the numbers of H, P, C control packets and TX-RX data-packets
%managed by the orcherstrator
fignumber = fignumber + 1;
figure(fignumber);
%
X1 = categorical({'Halt-pk', 'Pause-pk','Clock-pk','Sent-pk','Received-pk'});
Y1 = [aux1, aux2, aux3, aux4, aux5];
b = bar(X1, Y1, 0.1,'grouped','FaceColor','flat');
%
b.CData(1,:)= [1,0,0];%Red color
b.CData(2,:)=[0,0,1]; %Blu color
b.CData(3,:)= [0,1,0];%Green color
b.CData(4,:)=[1,0,1]; %RB color
b.CData(5,:)= [0,1,1];%GB color
%Set the colors of the two plotted bars
%
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
%Set the label at the tip of each plotted bar
xlabel(''); ylabel('Number of packets');
grid on;
title('FedE - Numbers of packets managed by the Orchestrator'); 
%
%Bar plots of the per-coworker numbers of D_packets actually processed by
%the orchestrator
fignumber = fignumber+1;
figure(fignumber);
b = bar([1:1:coworker_number],aux_vec3,0.02,'grouped','FaceColor','flat');
for j=1:size(aux_vec3,2)
    b.CData(j,:)= [(j/(size(aux_vec3,2))), 0.01*(j/(size(aux_vec3,2))), 0.01*(j/(size(aux_vec3,2)))];
    %Set the individual per-bar colors
end
xtips1 = b.XEndPoints;
ytips1 = b.YEndPoints;
labels1 = string(b.YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Number of processed packets');
title('FedE - Per-coworker numbers of data-packets processed by the Orchestrator'); 
%
%Bar plot of the propulsion energy consumed by the simulated RW-drone
if((FL_environment_flag == 1)&&(training_wall_clock_time< Inf)) %The simulated FL environment is RW-drone supported and the FL training ended without failure
    fignumber = fignumber+1;
    figure(fignumber);
    RW_UAV_consumed_propulsion_energy = compute_RW_UAW_propulsion_energy(gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, UAV_R, UAV_angular_speed, training_wall_clock_time);
    %Propulsion energy (in (Joule)) wasted by the supporting RW-drone during the FL training
    X1 = categorical({'RW-drone propulsion energy'});
    Y1 = round(RW_UAV_consumed_propulsion_energy*0.001,2);
    b = bar(X1, Y1, 0.1,'FaceColor','flat');
    %
    b.CData(1,:)= [1,0,1];
    %Set the color of the plotted bar
    grid on; 
    title('FedE - Propulsion energy (kiloJoule) wasted by the flying RW-drone hosting the orchestrator'); 
    ylabel('Propulsion energy (kJ)');
end% End if
%
%
%Bar plot of the propulsion power consumed by the emulated drone
if((FL_environment_flag == 1)&&(training_wall_clock_time < Inf)) %The simulated FL environment is RW-drone supported and the FL training ended without failure
    fignumber = fignumber+1;
    figure(fignumber);
    RW_UAV_consumed_propulsion_power = RW_UAV_consumed_propulsion_energy/training_wall_clock_time;
    %Propulsion power (in (W)) wasted by the supporting drone during the FL training
    X1 = categorical({'Drone propulsion power'});
    Y1 = round(RW_UAV_consumed_propulsion_power,2);
    b = bar(X1, Y1, 0.1,'FaceColor','flat');
    %
    b.CData(1,:)= [0,0,1];
    %Set the color of the plotted bar
    grid on; 
    title('FedE - Propulsion Power'); 
    ylabel('Propulsion power (Watt)');
end %End if
%
%
%Bar plots of the per-coworker computing, network and total energies
fignumber = fignumber+1;
figure(fignumber);
b = bar([1:1:coworker_number],aux_mtx4,0.11,'stacked');
xtips1 = b.XEndPoints;
ytips1 = max(b.YEndPoints);
labels1 = string(Total_energy_vec);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Consumed energy (Joule)');
title('FedE - Per-coworker computing, network and total energies'); 
legend({sprintf('Computing energy'),sprintf('Network energy')},'Location','NorthEast'); 
%
%Bar plots of the per-coworker computing, network and total consumed powers
%averaged over the overall FL training wall-clock duration
fignumber = fignumber+1;
figure(fignumber);
b = bar([1:1:coworker_number],round(1000*(aux_mtx4/(final_clock)),2), 0.02,'stacked');
xtips1 = b.XEndPoints;
ytips1 = max(b.YEndPoints);
labels1 = string(round(1000*(Total_energy_vec/(final_clock)),2));
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
grid on; xlabel('Coworker index'); ylabel('Average consumed power (mW)');
title('FedE - Per-coworker computing, network and total consumed powers averaged over the FL training wall-clock duration'); 
legend({sprintf('Average computing power'),sprintf('Average network power')},'Location','NorthEast');
%
%
%Plot of the sets of the global iteration indexes at which gradient 
% perturbations are applied
fignumber = fignumber +1;
figure(fignumber);
%
for j=1:1:size(perturbation_index_vec,2)
    if(perturbation_index_vec(j) == 1)
       plot(j, perturbation_index_vec(j), 'r*');
       hold on;
    else
       plot(j, perturbation_index_vec(j), 'r*');    
       hold on
    end %End if
end %End for
xlabel('Integer-valued global iteration index'); ylabel('ON/OFF perturbation');
grid on;
title('FedE - Binary plot of the global iteration indexes at which gradient perturbation is applied'); 
legend({sprintf('1 = Perturbation-ON'),sprintf('0 = Perturbation-OFF') },'Location','NorthEast'); 
%
%
%Bar plots of the returned maximum/minimum values of the global objective 
%function  
fignumber=fignumber+1;
figure(fignumber);
min_str = ['Minimum'];
%In the case of multiple maximum points, only the
%(1xprimal_vec_length)-dimensional coordinates of the FIRST maximum point
%are displayed

max_str = ['Maximum'];
%In the case of multiple minimum points, only the
%(1xprimal_vec_length)-dimensional coordinates of the FIRST minimum point
%are displayed

X1 = categorical({max_str, min_str});
Y1 = [round(or_gl_max_ob_function_aux,3), round(or_gl_min_ob_function_aux,3)];
b = bar(X1, Y1, 0.1,'grouped','FaceColor','flat');
%
b.CData(1,:)= [1,0,0];%Red color
b.CData(2,:)=[0,1,0]; %Green color
%Set the colors of the two plotted bars
%
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1, ytips1, labels1, 'HorizontalAlignment','center', 'VerticalAlignment','bottom');
%Set the label at the tip of each plotted bar
grid on; 
ylabel('Global maximum and minimum values');
title('FedE - Global objective function''s maximum and minimum');
%
%------------------End rendering section----------------------------------
%
fprintf('Rendering_fun ends succesfully\n');

%End Rendering_fun