
def get_variables( file ):

    result = {}

    ######### NuE_bfr Variables #########
    PrimaryNeutrinoEnergy_NuE_bfr = NuE_bfr_file['I3MCWeightDict']['PrimaryNeutrinoEnergy'].values
    PrimartNeutrinoType_NuE_bfr = NuE_bfr_file['I3MCWeightDict']['PrimaryNeutrinoType'].values
    PrimaryNeutrinoAzimuth_NuE_bfr = NuE_bfr_file['I3MCWeightDict']['PrimaryNeutrinoAzimuth'].values
    PrimaryNeutrinoZenith_NuE_bfr = NuE_bfr_file['I3MCWeightDict']['PrimaryNeutrinoZenith'].values
    MCInteractionType_NuE_bfr = NuE_bfr_file['I3MCWeightDict']['InteractionType'].values
    #MCInteractionDepth_NuE_bfr = NuE_bfr_file['penetrating_depth'].value.values

    TrueAzimuth_NuE_bfr = NuE_bfr_file['TrueAzimuth'].value.values
    TrueETot_NuE_bfr = NuE_bfr_file['TrueETot'].value.values
    TrueL_NuE_bfr = NuE_bfr_file['TrueL'].value.values
    TrueZenith_NuE_bfr = NuE_bfr_file['TrueZenith'].value.values
    TrueInteractionEventclass_NuE_bfr = NuE_bfr_file['MCInteractionEventclass'].value.values

    RecoAzimuth_NuE_bfr = NuE_bfr_file['RecoAzimuth'].value.values
    RecoEConfinement_NuE_bfr = NuE_bfr_file['RecoEConfinement'].value.values
    RecoERatio_NuE_bfr = NuE_bfr_file['RecoERatio'].value.values
    RecoZenith_NuE_bfr = NuE_bfr_file['RecoZenith'].value.values
    RecoL_NuE_bfr = NuE_bfr_file['RecoL'].value.values
    RecoETot_NuE_bfr = NuE_bfr_file['RecoETot'].value.values
    FinalEventClass_NuE_bfr = NuE_bfr_file['FinalTopology'].value.values
    ConventionalSelfVetoWeight_NuE_bfr = NuE_bfr_file['ConventionalAtmosphericPassingFractions'].value.values
    PromptSelfVetoWeight_NuE_bfr = NuE_bfr_file['PromptAtmosphericPassingFractions'].value.values



def error_cal(bin_edges,weights,data):
    errors = []
    bin_centers = []
    
    for bin_index in range(len(bin_edges) - 1):

        # find which data points are inside this bin
        bin_left = bin_edges[bin_index]
        bin_right = bin_edges[bin_index + 1]
        in_bin = np.logical_and(bin_left < data, data <= bin_right)
        

        # filter the weights to only those inside the bin
        weights_in_bin = weights[in_bin]

        # compute the error however you want
        error = np.sqrt(np.sum(weights_in_bin ** 2))
        errors.append(error)

        # save the center of the bins to plot the errorbar in the right place
        bin_center = (bin_right + bin_left) / 2
        bin_centers.append(bin_center)

    errors=np.asarray(errors)
    bin_centers=np.asarray(bin_centers)
    return errors, bin_centers


def plot_2dHist(x,y,weights,xbins_start,xbins_stop,xbins,ybins_start,ybins_stop,ybins,\
                xlogspace,ylogspace,title,eventcount,ax):
    import matplotlib as mat
    import matplotlib.font_manager as font_manager
    
    
    if xlogspace:
        x_bins=np.logspace(xbins_start,xbins_stop,xbins)
        
    else:
        x_bins=np.linspace(xbins_start,xbins_stop,xbins)
        
    
    if ylogspace:
        
        y_bins =np.logspace(ybins_start,ybins_stop,ybins)
    else:
        
        y_bins =np.linspace(ybins_start,ybins_stop,ybins)
    bins = [x_bins,y_bins]
    
    H, xedges, yedges = np.histogram2d(x,y,bins = [x_bins,y_bins],\
                                   weights=weights)
    H /= np.sum(H)
    
    
    #norm=colors.LogNorm()
    h = ax.pcolormesh(xedges, yedges, H.T,norm = mat.colors.LogNorm())
    #h = ax.pcolormesh(xedges, yedges, H.T,norm = mat.colors.Normalize(vmin=vmin, vmax=vmax))
    #h = ax.pcolormesh(xedges, yedges, H.T,norm = mat.colors.Normalize())
    ax.set_xlim(min(x_bins),max(x_bins))
    ax.set_ylim(min(y_bins),max(y_bins))
    ax.set_title(title,fontdict=font_axis_label)
    
    
    
    if xlogspace:
        ax.set_xscale('log')
        
        
    if ylogspace:
        ax.set_yscale('log')
        
        
        
    if eventcount:
        EventCount, _, __ = np.histogram2d(x,y,bins = [x_bins,y_bins])
        print("Total Hist Count is %d"%np.sum(EventCount))
        if xlogspace:
                midbins_x = np.sqrt(x_bins[:-1] * x_bins[1:])
                
        else:
                midbins_x = (x_bins[1:] - x_bins[:-1])/2 + x_bins[:-1]   
                
                
        if ylogspace:
                
                midbins_y = np.sqrt(y_bins[:-1] * y_bins[1:])
        else:
                
                midbins_y = (y_bins[1:] - y_bins[:-1])/2 + y_bins[:-1]   
                
        for i_x in range(len(xedges)-1):
            for i_y  in range(len(yedges)-1):
                ax.text(midbins_x[i_x],
                        midbins_y[i_y],
                        int(EventCount[i_x, i_y]),
                        color='w',
                        ha='center',
                        va='center',
                        fontweight='normal',
                        fontsize = 12,
                       )
   
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)
        item.set_family('serif')
    
    clb = plt.colorbar(h)
    
    clb.ax.tick_params(labelsize=16)
    
    
    