// Geometry: coaxial capacitor (circle with hole in the middle)
R1 = 0.2;  // radius of the inner circle
R2 = 1.0;  // radius of the outer circle

// Points for the inner circle
Point(1) = {0, 0, 0};  // center
Point(2) = {R1, 0, 0};
Point(3) = {0, R1, 0};
Point(4) = {-R1, 0, 0};
Point(5) = {0, -R1, 0};

// Points for the outer circle
Point(6) = {R2, 0, 0};
Point(7) = {0, R2, 0};
Point(8) = {-R2, 0, 0};
Point(9) = {0, -R2, 0};

// Arcs for the inner circle
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

// Arcs for the outer circle
Circle(5) = {6, 1, 7};
Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};
Circle(8) = {9, 1, 6};

// Loop for the inner circle
Line Loop(9) = {1, 2, 3, 4};

// Loop for the outer circle
Line Loop(10) = {5, 6, 7, 8};

// Surface (domain between circles)
Plane Surface(11) = {10, 9};  // outer circle minus inner circle

// Physical groups
Physical Surface("domain") = {11};
Physical Curve("inner") = {1, 2, 3, 4};  // inner circle
Physical Curve("outer") = {5, 6, 7, 8};  // outer circle

// Mesh parameters
Mesh.CharacteristicLengthMin = 0.01;
Mesh.CharacteristicLengthMax = 0.05;
