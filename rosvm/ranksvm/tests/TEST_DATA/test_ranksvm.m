function ndcg = test_ranksvm()
  load ohsumed
  C = 1e-4; % Constant penalizing the training errors
  for i=1:5
    A = generate_constraints(Yall(i).Y);
    tic;
    w = ranksvm(Xall(i).X, A, C*ones(size(A,1),1));
    t = toc;
    ndcg(i) = compute_ndcg(Xall(i).Xt*w,Yall(i).Yt,10);
    fprintf('Fold %d, NDCG@10 = %f, Time = %fs\n',i,ndcg(i),t);
  end;

function ndcg = test_ranksvm_with_ms()
% Same as before but find the C on the validation set
  load ohsumed
  for i=1:5
    A = generate_constraints(Yall(i).Y);
    for j=1:7
      w(:,j) = ranksvm(Xall(i).X, A, 10^(j-5)*ones(size(A,1),1));
      ndcg_ms(j) = compute_ndcg(Xall(i).Xv*w(:,j),Yall(i).Yv,10);
    end;
    fprintf('C = %f, ndcg = %f\n',[10.^[-4:2]; ndcg_ms])
    [foo, k] = max(ndcg_ms);
    for j=1:10
      ndcg(i,j) = compute_ndcg(Xall(i).Xt*w(:,k),Yall(i).Yt,j);
    end;
    fprintf('Fold %d: ',i);
    fprintf('%f ',ndcg(i,:));
    fprintf('\n');
  end;
  fprintf('Average: ');
  fprintf('%f ',mean(ndcg));
  fprintf('\n');
      
  
  
% Build the sparse matrix of constraints
% Each row is constraint. If the p-th example is prefered to the
% q-th one, there is a +1 on column p and a -1 on column q.
function A = generate_constraints(Y)
  nq = length(Y);
  
  I=zeros(1e7,1); J=I; V=I; nt = 0;
  
  ind = 0;
  for i=1:nq
    ind = ind(end)+[1:length(Y{i})]';
    Y2 = Y{i};
    n = length(ind);
    [I1,I2] = find(repmat(Y2,1,n)>repmat(Y2',n,1));
    n = length(I1);
    I(2*nt+1:2*nt+2*n) = nt+[1:n 1:n]'; 
    J(2*nt+1:2*nt+2*n) = [ind(I1); ind(I2)];
    V(2*nt+1:2*nt+2*n) = [ones(n,1); -ones(n,1)];
    nt = nt+n;
  end;
  A = sparse(I(1:2*nt),J(1:2*nt),V(1:2*nt));    
  
function ndcg = compute_ndcg(Y,Yt,p)
  conv = [0 1 3];
  ind = 0;
  for i=1:length(Yt)
    ind = ind(end)+[1:length(Yt{i})];
    q = min(p,length(ind));
    disc = [1 log2(2:q)];
    [foo,ind2] = sort(-Yt{i});
    best_dcg = sum(conv(Yt{i}(ind2(1:q))+1) ./ disc) + eps;
    [foo,ind2] = sort(-Y(ind));
    ndcg(i) = sum(conv(Yt{i}(ind2(1:q))+1) ./ disc) / best_dcg;
  end; 
  ndcg = mean(ndcg);
  