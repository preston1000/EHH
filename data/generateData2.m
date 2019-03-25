function [ xTrain, yTrain, xTest, yTest ] = generateData2( numTrain, numTest )
%GENERATEDATA2 for n
    nu = 2;
    ny = 3;
    %% train data generation
    yTrain = zeros(numTrain, 1);
    ns1 = max(nu, ny);

    xTrain = unifrnd( -1 , 1, numTrain, 1 );
    
    for t = ns1+1 : numTrain
        yTrain( t ) = ( yTrain( t - 1 ) * yTrain( t - 2 ) * yTrain( t - 3 ) * xTrain( t - 2 ) * ( yTrain( t - 3 ) - 1 ) + xTrain( t - 1 ) ) / ( 1 + yTrain( t - 2 )^2 + yTrain( t - 3 )^2 ) + normrnd( 0, 0.13 );
    end

    %% test data generation
    yTest = zeros(numTest, 1);
    xTest = zeros(numTest, 1);
    t = (1 : 500)';
    xTest(1:500, 1) = sin( 2 * pi * t / 250) ;
    t = (501 : numTest)';
    xTest(501:end, 1) = 0.8 * sin( 2 * pi * t / 250 ) + 0.2 * sin( 2 * pi * t / 25);
    
    for t = ns1+1 : numTest
        yTest( t ) = ( yTest( t - 1 ) * yTest( t - 2 ) * yTest( t - 3 ) * xTest( t - 2 ) * ( yTest( t - 3 ) - 1 ) + xTest( t - 1 ) ) / ( 1 + yTest( t - 2 )^2 + yTest( t - 3 )^2 );
    end

end

