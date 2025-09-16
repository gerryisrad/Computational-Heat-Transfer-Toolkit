clear all;
close all;
clc;

%% Problem Parameters Definition
k = 50;             % Thermal conductivity (W/m·K)
elem_size = 0.05;   % Element edge size for mesh generation
showmesh = 'on';    % Mesh visualization toggle
generation = 1;     % Thermal generation activation flag
flux = 10000;       % Heat flux boundary condition (W/m²)

%% Geometry Creation and Mesh Generation
model = createpde;
geometryFromEdges(model,@crackg);
mesh = generateMesh(model,'GeometricOrder','linear','Hmax',elem_size);

% Mesh visualization with various labeling options
figure(1);
pdemesh(model);
title('Mesh');

figure(2);
pdegplot(model,'EdgeLabels','on');
title('Mesh with Edge Labels');

figure(3);
pdemesh(model,'NodeLabels','on');
title('Mesh With Node Labels');

figure(4);
pdemesh(model,'ElementLabels','on');
title('Mesh with Element Labels');

%% Boundary Conditions Specification
% Dirichlet boundary conditions (temperature constraints)
dirichlet_edges = [7 6 2 1];
dirichlet_values = [25 25 100 100];

% Neumann boundary conditions (heat flux constraints)
neumann_edges = [3 4 5 8];
neumann_values = [flux flux flux flux];

%% Data Structure Initialization
nodes = mesh.Nodes;             % Node coordinate array
connectivity = mesh.Elements;   % Element connectivity matrix
n_nodes = length(nodes(1,:));   % Total number of nodes
n_elem = length(connectivity(1,:)); % Total number of elements

K_global = zeros(n_nodes, n_nodes); % Global stiffness matrix
f_global = zeros(n_nodes, 1);       % Global forcing vector
K_elem = zeros(3,3,n_elem);         % Element stiffness matrices storage
f_elem = zeros(3,n_elem);           % Element forcing vectors storage

% Gauss quadrature points and weights for triangular elements
gp = [2/3 1/6;1/6 2/3;1/6 1/6];
w = [1/3 1/3 1/3]';
counter = 0;

%% Element Matrix Assembly
for i_e = 1:n_elem
    % Extract nodal coordinates for current element
    n_location = nodes(:,connectivity(:,i_e))';

    % Gauss point integration loop
    for i_g = 1:3
        % Current Gauss point coordinates
        eps = gp(i_g,1);
        eta = gp(i_g,2);

        % Shape function evaluation
        N1 = eps;
        N2 = eta;
        N3 = 1 - eps - eta;
        N  = [N1 N2 N3];

        % Shape function derivatives in natural coordinates
        dN1deps = 1;  dN2deps = 0;  dN3deps = -1;
        dN1deta = 0;  dN2deta = 1;  dN3deta = -1;
        dN_mat = [dN1deps, dN2deps, dN3deps; dN1deta, dN2deta, dN3deta];

        % Physical coordinates of Gauss point
        xg = n_location(1,1)*N1 + n_location(2,1)*N2 + n_location(3,1)*N3;
        yg = n_location(1,2)*N1 + n_location(2,2)*N2 + n_location(3,2)*N3;

        % Jacobian matrix computation
        Jac = dN_mat*n_location;
        detJ = det(Jac);
        Jacinv = inv(Jac);

        % Shape function derivatives in physical coordinates
        dNdphys = Jac\dN_mat;

        % Element stiffness matrix contribution
        K_elem(:,:,i_e) = K_elem(:,:,i_e) + (dNdphys')*k*dNdphys*detJ*(1/3);

        % Thermal generation term
        s = generation*(1000*xg - 5000*yg);

        % Element forcing vector contribution
        f_elem(i_g,i_e) = (N(i_g))*s*detJ*(1/3);
    end
end

%% Neumann Boundary Condition Implementation
for i_n = 1:length(neumann_edges)
    % Identify nodes and elements on Neumann boundary
    neu_nodes = findNodes(mesh,'region','edge',neumann_edges(i_n));
    neu_elements = findElements(mesh,'attached',neu_nodes);

    n_val_at_edge = neumann_values(i_n);

    % Process each boundary element
    for i_e = 1:length(neu_elements)
        % Element node indices
        nu1 = connectivity(1,neu_elements(i_e));
        nu2 = connectivity(2,neu_elements(i_e));
        nu3 = connectivity(3,neu_elements(i_e));

        % Identify boundary nodes
        boundary_node = [0,0,0];
        if(ismember(nu1,neu_nodes))
            boundary_node(1) = 1;
        end
        if(ismember(nu2,neu_nodes))
            boundary_node(2) = 1;
        end
        if(ismember(nu3,neu_nodes))
            boundary_node(3) = 1;
        end

        % Process elements with two boundary nodes
        if(sum(boundary_node) == 2)
            % Node coordinates
            nu1_coords = nodes(:,nu1);
            nu2_coords = nodes(:,nu2);
            nu3_coords = nodes(:,nu3);

            % Determine boundary edge and compute integration parameters
            if(boundary_node(1) == 1 && boundary_node(2) == 1)
                dist = sqrt((nu2_coords(1)-nu1_coords(1))^2 + (nu2_coords(2)-nu1_coords(2))^2);
                gp1 = [(sqrt(3)-1)/(2*sqrt(3)),(sqrt(3)+1)/(2*sqrt(3))];
                gp2 = [(sqrt(3)+1)/(2*sqrt(3)),(sqrt(3)-1)/(2*sqrt(3))];
            elseif(boundary_node(1) == 1 && boundary_node(3) == 1)
                dist = sqrt((nu3_coords(1)-nu1_coords(1))^2 + (nu3_coords(2)-nu1_coords(2))^2);
                gp1 = [(sqrt(3)-1)/(2*sqrt(3)),0];
                gp2 = [(sqrt(3)+1)/(2*sqrt(3)),0];
            else
                dist = sqrt((nu3_coords(1)-nu2_coords(1))^2 + (nu3_coords(2)-nu2_coords(2))^2);
                gp1 = [0,(sqrt(3)-1)/(2*sqrt(3))];
                gp2 = [0,(sqrt(3)+1)/(2*sqrt(3))];
            end

            % Shape function evaluation at boundary Gauss points
            N1_g1 = gp1(1); N2_g1 = gp1(2); N3_g1 = 1-gp1(1)-gp1(2);
            N1_g2 = gp2(1); N2_g2 = gp2(2); N3_g2 = 1-gp2(1)-gp2(2);
            N_g1 = [N1_g1; N2_g1; N3_g1];
            N_g2 = [N1_g2; N2_g2; N3_g2];

            % Add boundary contribution to forcing vector
            f_elem(:,neu_elements(i_e)) = f_elem(:,neu_elements(i_e)) + (N_g1+N_g2)*(dist)*n_val_at_edge;
        end
    end
    counter = counter + 1;
end

%% Global System Assembly
for i_e = 1:n_elem
    n1 = connectivity(1,i_e);
    n2 = connectivity(2,i_e);
    n3 = connectivity(3,i_e);

    % Assemble element stiffness matrix into global matrix
    K_global(n1,n1) = K_global(n1,n1) + K_elem(1,1,i_e);
    K_global(n1,n2) = K_global(n1,n2) + K_elem(1,2,i_e);
    K_global(n1,n3) = K_global(n1,n3) + K_elem(1,3,i_e);
    K_global(n2,n1) = K_global(n2,n1) + K_elem(2,1,i_e);
    K_global(n2,n2) = K_global(n2,n2) + K_elem(2,2,i_e);
    K_global(n2,n3) = K_global(n2,n3) + K_elem(2,3,i_e);
    K_global(n3,n1) = K_global(n3,n1) + K_elem(3,1,i_e);
    K_global(n3,n2) = K_global(n3,n2) + K_elem(3,2,i_e);
    K_global(n3,n3) = K_global(n3,n3) + K_elem(3,3,i_e);

    % Assemble element forcing vector into global vector
    f_global(n1) = f_global(n1) + f_elem(1,i_e);
    f_global(n2) = f_global(n2) + f_elem(2,i_e);
    f_global(n3) = f_global(n3) + f_elem(3,i_e);
end

%% Dirichlet Boundary Condition Implementation
for i_edge = 1:length(dirichlet_edges)
    % Identify nodes on Dirichlet boundary
    edge_nodes = findNodes(mesh,'region','Edge',dirichlet_edges(i_edge));
    d_val_at_edge = dirichlet_values(i_edge);

    % Apply constraint to each boundary node
    for i_n = 1:length(edge_nodes)
        nodal_ID = edge_nodes(i_n);

        % Modify global system for constraint
        K_global(nodal_ID,:) = zeros(1,n_nodes);
        K_global(nodal_ID, nodal_ID) = 1;
        f_global(edge_nodes(i_n)) = d_val_at_edge;
    end
end

%% Solution
T_sol = K_global\f_global;

%% Results Visualization
figure(5);
pdeplot(model,'XYData',T_sol,'Mesh',showmesh,'ColorMap','jet');
title('Computed Temperature Field');

%% Reference Solution and Error Analysis
thermalmodel = createpde('thermal','steadystate');
geometryFromEdges(thermalmodel,@crackg);

thermalBC(thermalmodel,'edge',[1,2],'Temperature',100);
thermalBC(thermalmodel,'edge',[6,7],'Temperature',25);
thermalBC(thermalmodel,'edge',[3,4,5,8],'HeatFlux',flux);

thermalProperties(thermalmodel,'ThermalConductivity',k);

func = @(location,state)generation*(1000.*(location.x) - 5000.*(location.y));
internalHeatSource(thermalmodel,func);

thermalmesh = generateMesh(thermalmodel,'GeometricOrder','linear','Hmax',elem_size);
reference_result = solve(thermalmodel);

figure(6);
pdeplot(thermalmodel,'XYData',reference_result.Temperature,'Mesh',showmesh,'ColorMap','jet');
title('Reference Temperature Field');

error = reference_result.Temperature - T_sol;
total_error = sqrt(dot(error, error))

