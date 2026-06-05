function [Sbar,labels,E,S,A,Z,converge_Z,converge_Z_G] = Train_new2(X, cls_num, anchor,alpha,gamma,delta)
% X is a cell data, each cell is a matrix in size of d_v *N,each column is a sample;
% cls_num is the clustering number 
% anchor is the anchor number
% alpha,gamma and delta are the parameters
%%nips 只输入输出的维度交换
nV = length(X);
N = size(X{1},2);
%t=N;%%11.19
nC=cls_num;
t=anchor; %%11.19

%% ============================ Initialization ============================
for k=1:nV
    Z{k} = zeros(t,N); 
    W{k} = zeros(t,N);
    J{k} = zeros(t,N);
    S{k}  = ones(size(X{k},1),N);
    A{k} = zeros(size(X{k},1),t); %11.19
    E{k} = X{k}-A{k}*Z{k};%X{k}-A{k}*Z{k};
    Y{k} = zeros(size(X{k},1),N); %Y{2} = zeros(size(X{k},1),N);
end
w = zeros(t*N*nV,1);
j = zeros(t*N*nV,1);
sX = [t, N, nV];

Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 0.0001; max_mu = 10e12; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;
%0.0001
converge_Z=[];
converge_Z_G=[];
%ACCmax=[];
while(Isconverg == 0)
%============================== Upadate S^k ,i.e.W^k=============================
for k =1:nV
    S_linshi{k} = -(0.5*alpha*(E{k}.^2))/(gamma);
    S{k} = zeros(size(S_linshi{k}));
    for ii = 1:size(X{k},2)
        S{k}(:,ii) = EProjSimplex(S_linshi{k}(:,ii));
    end
    %S{k}=S_linshi{k};
end
   %% =========================== Upadate E^k, Y^k ===========================
 for k =1:nV
     GG{k}=X{k}-A{k}*Z{k}+Y{k}/mu;
      E{k} = (mu*GG{k})./(mu+alpha*S{k});
 end

%% ============================== Upadate Z^k =============================
     clear i
     sum_Z = zeros(t,N); %temp_E =[];
      for k =1:nV
          tmp = A{k}'*Y{k} + mu*A{k}'*X{k} +  rho*J{k} - W{k} -mu*A{k}'*E{k};
          Z{k}=inv(rho*eye(t,t)+ mu*eye(t,t))*tmp;
          for ii = 1:size(X{k},2)
              Z{k}(:,ii) = EProjSimplex_new(Z{k}(:,ii));
          end
          %- gama*sum_Z
          %temp_E=[temp_E,X{k}-Z{k}*A{k}+Y{k}/mu];
      end
      clear k 
      %% ============================= Upadate J^k ==============================
                Z_tensor = cat(3, Z{:,:});%%把所有的矩阵堆叠为张量 n*n*V
                W_tensor = cat(3, W{:,:});
                z = Z_tensor(:);
                w = W_tensor(:);
                J_tensor = solve_GG(Z_tensor + 1/rho*W_tensor,rho,sX,delta);
                j = J_tensor(:);
                %TNN
                % [j,objV] = wshrinkObj(Z_tensor + 1/rho*W_tensor,1/rho,sX,0,3);
                % J_tensor=reshape(j, sX);
%% ============================== Upadate A{v} ===============================
   G={};
for i = 1 :nV
    G{i}=(Y{i}+mu*(X{i})-mu*E{i})*Z{i}';
    [Au,ss,Av] = svd(G{i},'econ');
    A{i}=Au*Av';
end

for i=1:nV
    Y{i} = Y{i} + mu*(X{i}-A{i}*Z{i}-E{i});
end
%% ============================== Upadate W ===============================
        w = w + rho*(z - j);
        W_tensor = reshape(w, sX);
    for k=1:nV
        W{k} = W_tensor(:,:,k);
    end
%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-A{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-A{k}*Z{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-J{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-J{k},inf);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];

    if (iter>20)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end

Sbar=[];
for i = 1:nV
    Sbar=cat(1,Sbar,1/sqrt(nV)*Z{i});
end
[U,Sig,V] = mySVD(Sbar',nC); 


rand('twister',5489)
labels=litekmeans(U, nC, 'MaxIter', 100,'Replicates',10);

%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');