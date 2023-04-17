// STAGE 4*: mehrere Photonen von 2-200GeV

// hauseigene Klassen
#include "Phast.h" 
#include "TTree.h"  
#include "TH2.h"
// allgemeine Klassen
#include <iostream> 
#include <vector>
#include <cmath>

void UserEvent2023048(PaEvent& e){
    static bool first = true;
    static TTree *tree = new TTree("user202302", "ECALinfo");
    // Parameter aus Fit (jetzt Vektroen, da mehrere Fits)
    static vector<double> *Efit = new std::vector<double>(0);
    static vector<double> *xfit = new std::vector<double>(0);
    static vector<double> *yfit = new std::vector<double>(0);
    static vector<double> *chi2fit = new std::vector<double>(0);
    static vector<double> *ndffit = new std::vector<double>(0);
    static int num_fit;
    // clearen, damit nicht jedes Mal neu befuellt
    Efit -> clear();
    xfit -> clear();
    yfit -> clear();
    chi2fit -> clear();
    ndffit -> clear();
    // Parameter aus MC Messung -> Werte fuer jede Zelle!
    static vector<float> *EMC = new std::vector<float>(0);
    static vector<double> *xMC = new std::vector<double>(0);
    static vector<double> *yMC = new std::vector<double>(0);
    static vector<int> *celltype = new std::vector<int>(0);
    static vector<int> *i_col = new std::vector<int>(0);
    static vector<int> *i_row = new std::vector<int>(0);
    // static vector<int> *num_hitcells = new std::vector<int>(0); // wie viele Cellen ge
    static vector<double> *momentum = new std::vector<double>(0);
    // damit nicht jedes Mal neu befuellt, da push back
    EMC->clear();
    xMC->clear();
    yMC->clear();
    momentum ->clear();
    celltype->clear();
    i_col->clear();
    i_row->clear();
    // Parameter aus MC truth (Vektor mit zwei Parametern, da zwei Photonen!)
    static vector<double> *Etruth= new std::vector<double>(0);
    static vector<double> *xtruth= new std::vector<double>(0);
    static vector<double> *ytruth= new std::vector<double>(0);
    //clearen
    Etruth -> clear();
    xtruth -> clear();
    ytruth  -> clear();
    // zaehle wie viele shower coral auf einen cluster fitted:
    int N_ecal2_cluster = 0;
    // position ecal oberflaeche
    double z_end = 3302.7; //cm
    // Abstand ecal zu Photon Start
    double del_z; 
    // speichere keine Zellen doppelt, nur coral nur einen Fit macht
    static vector<int> *index_shower = new std::vector<int>(0);
    index_shower -> clear();

    if(first){
        first = false;
        tree -> Branch("E_fit", &Efit);
        tree -> Branch("x_fit", &xfit);
        tree -> Branch("y_fit", &yfit);
        tree -> Branch("chi2_fit", &chi2fit);
        tree -> Branch("ndf_fit", &ndffit);
        tree -> Branch("num_fit", &num_fit);
        tree -> Branch("E_MC", &EMC);
        tree -> Branch("x_MC", &xMC);
        tree -> Branch("y_MC", &yMC);
        tree -> Branch("celltype", &celltype);
        tree -> Branch("irow", &i_col);
        tree -> Branch("icol", &i_row);
        tree -> Branch("E_truth", &Etruth);
        tree -> Branch("x_truth", &xtruth);
        tree -> Branch("y_truth", &ytruth);
        tree -> Branch("momentum", &momentum);
    }

    // finde MC truth Parameter     -   aka erster Vertex, an dem Photon simuliert wurde 

    for(int i=0; i<e.NMCvertex(); ++i){
        const PaMCvertex &vertex = e.vMCvertex(i);
        if(!vertex.IsPrimary()) continue;
        double z_start = vertex.Pos(2); // wo startet photon (3300cm)
        del_z = z_end - z_start;
        if(vertex.NMCtrack() == 2){ // zwei outgoing photonen, die sich hoffentlich dann ueberlappen :-) 
            for(int j=0; j<2; ++j){
                int ind_photontrack = vertex.iMCtrack(j);
                const PaMCtrack &photontrack = e.vMCtrack(ind_photontrack);
                if(photontrack.E()<1) return; // breche ab, falls photon sehr kleine Energie (1GeV) hat
                Etruth -> push_back(photontrack.E());
                // winkelabhaengige x und y Position! 
                double del_x = photontrack.Px() / photontrack.Pz() * del_z;
                double del_y = photontrack.Py() / photontrack.Pz() * del_z;
                xtruth -> push_back(vertex.Pos(0) + del_x); 
                ytruth -> push_back(vertex.Pos(1) + del_y);
                // ueberpruefe ob photon durchs Loch geht -> falls ja, breche ab!
                if(((vertex.Pos(0) + del_x) > 10 and (vertex.Pos(0) + del_x) < 19) and ((vertex.Pos(1) + del_y) > (-4) and (vertex.Pos(1) + del_y) < 4)) return;
                // speichere Impuls
                momentum -> push_back(photontrack.Px());
                momentum -> push_back(photontrack.Py());
                momentum -> push_back(photontrack.Pz());
                // ueberpruefe ob photon bei ecal2 ankommt!
                const vector<int>& out_vertices = photontrack.vMCvertex();
                if(!out_vertices.empty()) return; // breche ab, falls Interaction stattfindet
            }
        } else{
            std::logic_error("Primary vertex should only have two outgoing photons!");
        }
    }

    // finde Parameter aus Fit

    const PaCalorimeter &ecal2 = PaSetup::Ref().Calorimeter(1);
    static double x_ECAL = ecal2.X();
    static double y_ECAL = ecal2.Y();
    const vector<PaCalorimCell> &calocells = ecal2.vCalorimCell();

    // loope ueber alle CAL clusters:
    for(int i=0; i<e.NCaloClus(); ++i){
        const PaCaloClus &cluster = e.vCaloClus(i);

        // nur vertices von ecal2
        if(!(cluster.iCalorim() == 1)) continue; // ab hier nur zwei mal!
        N_ecal2_cluster++; // wie viele Fits?

        // Positionskorrektur Coral
        const float zfit = cluster.Z();
        double cdel_z = z_end - zfit; // Entfernung ecal oberflaeche zu coral fit z, "Ende minus Anfang vom Extropolieren"
        double cdel_x = cdel_z * momentum->at(0) / momentum->at(2);
        double cdel_y = cdel_z * momentum->at(1) / momentum->at(2);
        
        xfit -> push_back(cluster.X() + cdel_x); // Korrektur zur z Ebene von ECAL Anfang
        yfit -> push_back(cluster.Y() + cdel_y); // es gibt auch Fehler vom Fit... vielleicht fuer spaeter?
        Efit -> push_back(cluster.E());
        chi2fit -> push_back(cluster.Chi2());
        ndffit -> push_back(cluster.Ndf());

        // ueberpruefe ob Zellenwerte schon mal gespeichert wurden, weil coral falsche anahl von fits macht...
        bool cont = true;
        for(unsigned int j=0; j<index_shower->size(); ++j){
            if(cluster.MyIndex() == index_shower -> at(j)) cont = false;
        }
        index_shower -> push_back(cluster.MyIndex());
        // finde Parameter aus MC Messung -> Werte je Zelle -> Fuelle Vektoren
        const vector<int> &cluster_cellnumbers = cluster.vCellNumber();
        if(cont == true){ // speichere nur einmal Cluster infos, auch wenn Coral falsche Anzahl an Shower fittet
            for(unsigned int j=0; j<cluster_cellnumbers.size(); ++j){
                const PaCalorimCell &onecell = calocells.at(cluster_cellnumbers.at(j));
                xMC -> push_back(onecell.Position().X() + x_ECAL);
                yMC -> push_back(onecell.Position().Y()+ y_ECAL);
                EMC -> push_back(cluster.vCellEnergy().at(j));
                int ecaltype;
                if(onecell.iType()<=11) ecaltype = 1; // gams
                if(onecell.iType()>=12 && onecell.iType()<=23) ecaltype = 2; //gams-r
                if(onecell.iType()>=24 && onecell.iType()<=31) ecaltype = 3; // shashlik
                celltype -> push_back(ecaltype);
                // ueberpruefe ob Cluster am Rand liegt:
                int icol = onecell.iColumn();
                int irow = onecell.iRow();
                i_col -> push_back(icol);
                i_row -> push_back(irow);
                if(icol<=8 || icol>=55 || irow<=12 || irow>=35) return; // grob abscheiden       
                if(((irow>12 && irow<20) || (irow>27 && irow<35)) && (icol<17 || icol>46)) return; // ecken abschneiden
                if(((irow>12 && irow<17) || (irow>30 && irow<35)) && (icol<21 || icol>42)) return; // ecken abschneiden
                if(icol>=33 && icol<=36 && irow>=22 && irow<=25) return; // beamloch rausschneiden
            }
        }
    }

    num_fit = N_ecal2_cluster; // wichtig diesen Parameter in Python zu ueberpruefen!!!
    if(EMC -> size()!=0) tree->Fill(); // befuelle Tree falls ein schoener Cluster im Event :-) 
    
}

