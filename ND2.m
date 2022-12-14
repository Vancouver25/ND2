clear all;
close all;

% Aproximasion using back propogasion
x = 0.1:1/22:1; % input
d = (0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2; %desired response

w11 = randn(1);
b11 = randn(1);
w12 = randn(1);
b12 = randn(1);
w13 = randn(1);
b13 = randn(1);
w14 = randn(1);
b14 = randn(1);

w21 = randn(1);
w22 = randn(1);
w23 = randn(1);
w24 = randn(1);
b21 = randn(1);
e = 0;
e_total =0;
n = 0.015;
 
for index = 1:length(x)
% Calculate hidden layer values, activation f(x) is a sigmoid
y1 = 1 / ( 1 + exp( -(x(index) * w11 + b11 )));
y2 = 1 / ( 1 + exp( -(x(index) * w12 + b12 )));
y3 = 1 / ( 1 + exp( -(x(index) * w13 + b13 )));
y4 = 1 / ( 1 + exp( -(x(index) * w14 + b14 )));

% Add all of the hidden layer values, activation f(x) is linear
y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + b21;

% Calculate error
e(index) = d(index) - y(index);
end;

% Update the weights
for m = 1:1000000 % executes while the total error is not 0
 % Update weights from the end layer "Backpropogating"
 
     for index = 1:length(x)
     w21 = w21 + n * e(index) * y1;
     w22 = w22 + n * e(index) * y2;
     w23 = w23 + n * e(index) * y3;
     w24 = w24 + n * e(index) * y4;
     b21 = b21 + n * e(index);
     end;
     
  %Update hidden layer weights  
     for index = 1:length(x)
     %Gradient calculation
     gradient1 = 1 / (1 + exp( -(x(index) * w11 + b11))) * (1-( 1 / (1 + exp( -(x(index) * w11 + b11))))) * e(index) * w21;
     w11 = w11 + n * gradient1 * x(index);
     b11 = b11 + n * gradient1;
     
     gradient2 = 1 / (1 + exp( -(x(index) * w12 + b12))) * (1-( 1 / (1 + exp( -(x(index) * w12 + b12))))) * e(index) * w22;
     w12 = w12 + n * gradient2 * x(index);
     b12 = b12 + n * gradient2;
     
     gradient3 = 1 / (1 + exp( -(x(index) * w13 + b13))) * (1-( 1 / (1 + exp( -(x(index) * w13 + b13))))) * e(index) * w23;
     w13 = w13 + n * gradient3 * x(index);
     b13 = b13 + n * gradient3;
     
     gradient4 = 1 / (1 + exp( -(x(index) * w14 + b14))) * (1-( 1 / (1 + exp( -(x(index) * w14 + b14))))) * e(index) * w24;
     w14 = w14 + n * gradient4 * x(index);
     b14 = b14 + n * gradient4;
     end;
     
        for index = 1:length(x)
        % Calculate hidden layer values, activation f(x) is a sigmoid
        y1 = 1 / ( 1 + exp( -(x(index) * w11 + b11 )));
        y2 = 1 / ( 1 + exp( -(x(index) * w12 + b12 )));
        y3 = 1 / ( 1 + exp( -(x(index) * w13 + b13 )));
        y4 = 1 / ( 1 + exp( -(x(index) * w14 + b14 )));

        % Add all of the hidden layer values, activation f(x) is linear
        y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + b21;
        
        % Calculate error
        e(index) = d(index) - y(index);
        end;
end;    


%learning data set---------------------------------------------------------   
  % Solve equation with new values
        for index = 1:length(x)
        % Calculate hidden layer values, activation f(x) is a sigmoid
        y1 = 1 / ( 1 + exp( -(x(index) * w11 + b11 )));
        y2 = 1 / ( 1 + exp( -(x(index) * w12 + b12 )));
        y3 = 1 / ( 1 + exp( -(x(index) * w13 + b13 )));
        y4 = 1 / ( 1 + exp( -(x(index) * w14 + b14 )));
        
        % Add all of the hidden layer values, activation f(x) is linear
        Y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + b21;
        end;
    
figure('Name','Learning data set approximation result');
plot(x,d, 'b', x,Y, 'rx');
grid on;
legend('Original signal','Signal approximation'); 
    


%test data set 1---------------------------------------------------------
clear Y d;
x = 0.5:0.4/30:1; %input values
d = (0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2; %desired response
  % Solve equation with new values
        for index = 1:length(x)
        % Calculate hidden layer values, activation f(x) is a sigmoid
        y1 = 1 / ( 1 + exp( -(x(index) * w11 + b11 )));
        y2 = 1 / ( 1 + exp( -(x(index) * w12 + b12 )));
        y3 = 1 / ( 1 + exp( -(x(index) * w13 + b13 )));
        y4 = 1 / ( 1 + exp( -(x(index) * w14 + b14 )));
        
        % Add all of the hidden layer values, activation f(x) is linear
        Y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + b21;
        end;
    
figure('Name','Test data set 1 approximation result');
plot(x,d, 'b', x,Y, 'rx');
grid on;
legend('Original signal','Signal approximation'); %error max 0.05



%test data set 2---------------------------------------------------------
clear Y d;
x = 0.1:0.5/44:1; %input values
d = (0.6*sin(2*pi*x/0.7) + 0.3*sin(2*pi*x))/2; %desired response
  % Solve equation with new values
        for index = 1:length(x)
        % Calculate hidden layer values, activation f(x) is a sigmoid
        y1 = 1 / ( 1 + exp( -(x(index) * w11 + b11 )));
        y2 = 1 / ( 1 + exp( -(x(index) * w12 + b12 )));
        y3 = 1 / ( 1 + exp( -(x(index) * w13 + b13 )));
        y4 = 1 / ( 1 + exp( -(x(index) * w14 + b14 )));
        
        % Add all of the hidden layer values, activation f(x) is linear
        Y(index) = y1*w21 + y2*w22 + y3*w23 + y4*w24 + b21;
        end;
    
figure('Name','Test data set 2 approximation result');
plot(x,d, 'b', x,Y, 'rx');
grid on;
legend('Original signal','Signal approximation'); 